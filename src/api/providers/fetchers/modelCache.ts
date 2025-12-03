import * as path from "path"
import fs from "fs/promises"
import * as fsSync from "fs"

import NodeCache from "node-cache"
import crypto from "crypto"
import { z } from "zod"

import type { ProviderName } from "@roo-code/types"
import { modelInfoSchema, TelemetryEventName } from "@roo-code/types"
import { TelemetryService } from "@roo-code/telemetry"

import { safeWriteJson } from "../../../utils/safeWriteJson"

import { ContextProxy } from "../../../core/config/ContextProxy"
import { getCacheDirectoryPath } from "../../../utils/storage"
import type { RouterName, ModelRecord } from "../../../shared/api"
import { fileExistsAtPath } from "../../../utils/fs"

import { getOpenRouterModels } from "./openrouter"
import { getVercelAiGatewayModels } from "./vercel-ai-gateway"
import { getRequestyModels } from "./requesty"
import { getGlamaModels } from "./glama"
import { getUnboundModels } from "./unbound"
import { getLiteLLMModels } from "./litellm"
import { GetModelsOptions } from "../../../shared/api"
import { getOllamaModels } from "./ollama"
import { getLMStudioModels } from "./lmstudio"
import { getIOIntelligenceModels } from "./io-intelligence"
import { getDeepInfraModels } from "./deepinfra"
import { getHuggingFaceModels } from "./huggingface"
import { getRooModels } from "./roo"
import { getChutesModels } from "./chutes"

const memoryCache = new NodeCache({ stdTTL: 5 * 60, checkperiod: 5 * 60 })

// Zod schema for validating ModelRecord structure from disk cache
const modelRecordSchema = z.record(z.string(), modelInfoSchema)

// Track in-flight refresh requests to prevent concurrent API calls for the same provider+profile
// Use a composite string key (provider|baseUrl|apiKey) so separate profiles don't block each other
const inFlightRefresh = new Map<string, Promise<ModelRecord>>()

/**
 * Compute a cache key from provider options. Includes baseUrl and apiKey so
 * different profiles/configs do not share the same cache entry.
 */
function computeCacheKey(options: GetModelsOptions): string {
	const provider = options.provider
	const base = options.baseUrl ?? ""
	const key = `${provider}|${base}|${options.apiKey ?? ""}`
	return key
}

function computeHash(key: string) {
	return crypto.createHash("sha1").update(key).digest("hex")
}

async function writeModels(provider: RouterName, cacheKey: string, data: ModelRecord) {
	const hash = computeHash(cacheKey)
	const filename = `${provider}_${hash}_models.json`
	const cacheDir = await getCacheDirectoryPath(ContextProxy.instance.globalStorageUri.fsPath)
	await safeWriteJson(path.join(cacheDir, filename), data)
}

async function readModels(provider: RouterName, cacheKey: string): Promise<ModelRecord | undefined> {
	const hash = computeHash(cacheKey)
	const filename = `${provider}_${hash}_models.json`
	const cacheDir = await getCacheDirectoryPath(ContextProxy.instance.globalStorageUri.fsPath)
	const filePath = path.join(cacheDir, filename)
	const exists = await fileExistsAtPath(filePath)
	return exists ? JSON.parse(await fs.readFile(filePath, "utf8")) : undefined
}

/**
 * Internal helper: check memory cache then disk for a specific cacheKey.
 * This keeps profile-scoped cache separate from legacy provider-only cache.
 */
function getModelsFromCacheForKey(cacheKey: string, provider: ProviderName): ModelRecord | undefined {
	// Memory cache first
	const memoryModels = memoryCache.get<ModelRecord>(cacheKey)
	if (memoryModels) {
		return memoryModels
	}

	// Disk cache: synchronous read for callers that expect sync behavior
	try {
		const hash = computeHash(cacheKey)
		const filename = `${provider}_${hash}_models.json`
		const cacheDir = getCacheDirectoryPathSync()
		if (!cacheDir) return undefined

		const filePath = path.join(cacheDir, filename)
		if (fsSync.existsSync(filePath)) {
			const data = fsSync.readFileSync(filePath, "utf8")
			const models = JSON.parse(data)
			const validation = modelRecordSchema.safeParse(models)
			if (!validation.success) {
				console.error(`[MODEL_CACHE] Invalid disk cache data structure for ${provider} (profile-scoped):`, validation.error.format())
				return undefined
			}

			memoryCache.set(cacheKey, validation.data)
			return validation.data
		}
	} catch (error) {
		console.error(`[MODEL_CACHE] Error loading ${provider} models from disk (profile-scoped):`, error)
	}

	return undefined
}

/**
 * Fetch models from the provider API.
 * Extracted to avoid duplication between getModels() and refreshModels().
 *
 * @param options - Provider options for fetching models
 * @returns Fresh models from the provider API
 */
async function fetchModelsFromProvider(options: GetModelsOptions): Promise<ModelRecord> {
	const { provider } = options

	let models: ModelRecord

	switch (provider) {
		case "openrouter":
			models = await getOpenRouterModels()
			break
		case "requesty":
			// Requesty models endpoint requires an API key for per-user custom policies.
			models = await getRequestyModels(options.baseUrl, options.apiKey)
			break
		case "glama":
			models = await getGlamaModels()
			break
		case "unbound":
			// Unbound models endpoint requires an API key to fetch application specific models.
			models = await getUnboundModels(options.apiKey)
			break
		case "litellm":
			// Type safety ensures apiKey and baseUrl are always provided for LiteLLM.
			models = await getLiteLLMModels(options.apiKey, options.baseUrl)
			break
		case "ollama":
			models = await getOllamaModels(options.baseUrl, options.apiKey)
			break
		case "lmstudio":
			models = await getLMStudioModels(options.baseUrl)
			break
		case "deepinfra":
			models = await getDeepInfraModels(options.apiKey, options.baseUrl)
			break
		case "io-intelligence":
			models = await getIOIntelligenceModels(options.apiKey)
			break
		case "vercel-ai-gateway":
			models = await getVercelAiGatewayModels()
			break
		case "huggingface":
			models = await getHuggingFaceModels()
			break
		case "roo": {
			// Roo Code Cloud provider requires baseUrl and optional apiKey
			const rooBaseUrl = options.baseUrl ?? process.env.ROO_CODE_PROVIDER_URL ?? "https://api.roocode.com/proxy"
			models = await getRooModels(rooBaseUrl, options.apiKey)
			break
		}
		case "chutes":
			models = await getChutesModels(options.apiKey)
			break
		default: {
			// Ensures router is exhaustively checked if RouterName is a strict union.
			const exhaustiveCheck: never = provider
			throw new Error(`Unknown provider: ${exhaustiveCheck}`)
		}
	}

	return models
}

/**
 * Get models from the cache or fetch them from the provider and cache them.
 * There are two caches:
 * 1. Memory cache - This is a simple in-memory cache that is used to store models for a short period of time.
 * 2. File cache - This is a file-based cache that is used to store models for a longer period of time.
 *
 * @param router - The router to fetch models from.
 * @param apiKey - Optional API key for the provider.
 * @param baseUrl - Optional base URL for the provider (currently used only for LiteLLM).
 * @returns The models from the cache or the fetched models.
 */
export const getModels = async (options: GetModelsOptions): Promise<ModelRecord> => {
	const { provider } = options

	const cacheKey = computeCacheKey(options)

	let models = getModelsFromCacheForKey(cacheKey, provider)

	if (models) {
		return models
	}

	try {
		models = await fetchModelsFromProvider(options)
		const modelCount = Object.keys(models).length

		// Only cache non-empty results to prevent persisting failed API responses
		// Empty results could indicate API failure rather than "no models exist"
		if (modelCount > 0) {
			memoryCache.set(cacheKey, models)

			await writeModels(provider, cacheKey, models).catch((err) =>
				console.error(`[MODEL_CACHE] Error writing ${provider} models to file cache:`, err),
			)
		} else {
			TelemetryService.instance.captureEvent(TelemetryEventName.MODEL_CACHE_EMPTY_RESPONSE, {
				provider,
				context: "getModels",
				hasExistingCache: false,
			})
		}

		return models
	} catch (error) {
		// Log the error and re-throw it so the caller can handle it (e.g., show a UI message).
		console.error(`[getModels] Failed to fetch models in modelCache for ${provider}:`, error)

		throw error // Re-throw the original error to be handled by the caller.
	}
}

/**
 * Force-refresh models from API, bypassing cache.
 * Uses atomic writes so cache remains available during refresh.
 * This function also prevents concurrent API calls for the same provider using
 * in-flight request tracking to avoid race conditions.
 *
 * @param options - Provider options for fetching models
 * @returns Fresh models from API, or existing cache if refresh yields worse data
 */
export const refreshModels = async (options: GetModelsOptions): Promise<ModelRecord> => {
	const { provider } = options
	const cacheKey = computeCacheKey(options)

	// Check if there's already an in-flight refresh for this provider+profile
	const existingRequest = inFlightRefresh.get(cacheKey)
	if (existingRequest) {
		return existingRequest
	}

	// Create the refresh promise and track it
	const refreshPromise = (async (): Promise<ModelRecord> => {
		try {
			// Force fresh API fetch - skip getModelsFromCache() check
			const models = await fetchModelsFromProvider(options)
			const modelCount = Object.keys(models).length

			// Get existing cached data for comparison
			const existingCache = getModelsFromCacheForKey(cacheKey, provider)
			const existingCount = existingCache ? Object.keys(existingCache).length : 0

			if (modelCount === 0) {
				TelemetryService.instance.captureEvent(TelemetryEventName.MODEL_CACHE_EMPTY_RESPONSE, {
					provider,
					context: "refreshModels",
					hasExistingCache: existingCount > 0,
					existingCacheSize: existingCount,
				})
				if (existingCount > 0) {
					return existingCache!
				} else {
					return {}
				}
			}

			// Update memory cache first
			memoryCache.set(cacheKey, models)

			// Atomically write to disk (safeWriteJson handles atomic writes)
			await writeModels(provider, cacheKey, models).catch((err) =>
				console.error(`[refreshModels] Error writing ${provider} models to disk:`, err),
			)

			return models
		} catch (error) {
			// Log the error for debugging, then return existing cache if available (graceful degradation)
			console.error(`[refreshModels] Failed to refresh ${provider} models:`, error)
			return getModelsFromCacheForKey(cacheKey, provider) || {}
		} finally {
			// Always clean up the in-flight tracking
			inFlightRefresh.delete(cacheKey)
		}
	})()

	// Track the in-flight request
	inFlightRefresh.set(cacheKey, refreshPromise)

	return refreshPromise
}

/**
 * Initialize background model cache refresh.
 * Refreshes public provider caches without blocking or requiring auth.
 * Should be called once during extension activation.
 */
export async function initializeModelCacheRefresh(): Promise<void> {
	// Wait for extension to fully activate before refreshing
	setTimeout(async () => {
		// Providers that work without API keys
		const publicProviders: Array<{ provider: RouterName; options: GetModelsOptions }> = [
			{ provider: "openrouter", options: { provider: "openrouter" } },
			{ provider: "glama", options: { provider: "glama" } },
			{ provider: "vercel-ai-gateway", options: { provider: "vercel-ai-gateway" } },
			{ provider: "chutes", options: { provider: "chutes" } },
		]

		// Refresh each provider in background (fire and forget)
		for (const { options } of publicProviders) {
			refreshModels(options).catch(() => {
				// Silent fail - old cache remains available
			})

			// Small delay between refreshes to avoid API rate limits
			await new Promise((resolve) => setTimeout(resolve, 500))
		}
	}, 2000)
}

/**
 * Flush models memory cache for a specific router.
 *
 * @param router - The router to flush models for.
 * @param refresh - If true, immediately fetch fresh data from API
 */
export const flushModels = async (router: RouterName, refresh: boolean = false): Promise<void> => {
	// Remove any profile-scoped memory cache entries and legacy provider key
	try {
		const prefix = `${router}|`

		// Delete profile-scoped memory cache keys
		for (const key of memoryCache.keys()) {
			if (key === router || key.startsWith(prefix)) {
				memoryCache.del(key)
			}
		}

		// Also remove legacy disk files matching provider_*.json
		const cacheDir = await getCacheDirectoryPath(ContextProxy.instance.globalStorageUri.fsPath)
		try {
			const files = await fs.readdir(cacheDir)
			for (const f of files) {
				if (f.startsWith(`${router}_`) && f.endsWith(`_models.json`)) {
					await fs.unlink(path.join(cacheDir, f)).catch(() => {
						// ignore individual unlink errors
					})
				}
			}
		} catch (err) {
			// ignore if cache dir doesn't exist yet
		}
	} catch (err) {
		console.error(`[flushModels] Error clearing caches for ${router}:`, err)
	}

	if (refresh) {
		// Trigger a refresh for the provider default options (will populate a default cache)
		refreshModels({ provider: router } as GetModelsOptions).catch((error) => {
			console.error(`[flushModels] Refresh failed for ${router}:`, error)
		})
	}
}

/**
 * Get models from cache, checking memory first, then disk.
 * This ensures providers always have access to last known good data,
 * preventing fallback to hardcoded defaults on startup.
 *
 * @param provider - The provider to get models for.
 * @returns Models from memory cache, disk cache, or undefined if not cached.
 */
export function getModelsFromCache(provider: ProviderName): ModelRecord | undefined {
	// First, try to find any profile-scoped memory cache for this provider
	const prefix = `${provider}|`
	for (const key of memoryCache.keys()) {
		if (key === provider || key.startsWith(prefix)) {
			const memoryModels = memoryCache.get<ModelRecord>(key)
			if (memoryModels) return memoryModels
		}
	}

	// Fallback to legacy provider-only disk file for backward compatibility
	try {
		const filename = `${provider}_models.json`
		const cacheDir = getCacheDirectoryPathSync()
		if (!cacheDir) {
			return undefined
		}

		const filePath = path.join(cacheDir, filename)

		// Use synchronous fs to avoid async complexity in getModel() callers
		if (fsSync.existsSync(filePath)) {
			const data = fsSync.readFileSync(filePath, "utf8")
			const models = JSON.parse(data)

			// Validate the disk cache data structure using Zod schema
			// This ensures the data conforms to ModelRecord = Record<string, ModelInfo>
			const validation = modelRecordSchema.safeParse(models)
			if (!validation.success) {
				console.error(
					`[MODEL_CACHE] Invalid disk cache data structure for ${provider}:`,
					validation.error.format(),
				)
				return undefined
			}

			// Populate memory cache for future fast access under legacy key
			memoryCache.set(provider, validation.data)

			return validation.data
		}
	} catch (error) {
		console.error(`[MODEL_CACHE] Error loading ${provider} models from disk:`, error)
	}

	return undefined
}

/**
 * Synchronous version of getCacheDirectoryPath for use in getModelsFromCache.
 * Returns the cache directory path without async operations.
 */
function getCacheDirectoryPathSync(): string | undefined {
	try {
		const globalStoragePath = ContextProxy.instance?.globalStorageUri?.fsPath
		if (!globalStoragePath) {
			return undefined
		}
		const cachePath = path.join(globalStoragePath, "cache")
		return cachePath
	} catch (error) {
		console.error(`[MODEL_CACHE] Error getting cache directory path:`, error)
		return undefined
	}
}
