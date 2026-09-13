/**
 * SIH26066 — OceanEmbed Backend API Client
 * Centralized service for communicating with the real FastAPI backend.
 */

import {
  HealthResponse,
  ModelInfoResponse,
  AvailableDatesResponse,
  PredictionRequest,
  PredictionResponse,
} from '../types';

const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  // Automatically adapt to the host accessing the frontend (e.g. 192.168.x.x or localhost)
  if (typeof window !== 'undefined' && window.location.hostname) {
    return `http://${window.location.hostname}:8000`;
  }
  return 'http://127.0.0.1:8000';
};

const API_BASE_URL = getApiBaseUrl();

class ApiError extends Error {
  statusCode: number;
  details?: any;

  constructor(message: string, statusCode: number, details?: any) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

export const api = {
  baseUrl: API_BASE_URL,

  /**
   * Checks whether the FastAPI backend is running and model is loaded.
   */
  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new ApiError(`Health check failed with status ${response.status}`, response.status);
      }

      return await response.json();
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError('Unable to connect to OceanEmbed backend server at ' + API_BASE_URL, 0, err);
    }
  },

  /**
   * Retrieves verified model architecture specifications and domain metadata.
   */
  async getModelInfo(): Promise<ModelInfoResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/model-info`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new ApiError(`Failed to fetch model info (${response.status})`, response.status);
      }

      return await response.json();
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError('Error contacting model-info endpoint', 0, err);
    }
  },

  /**
   * Retrieves the list of supported observation dates (Jan-Sep 2020).
   */
  async getAvailableDates(): Promise<AvailableDatesResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/available-dates`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new ApiError(`Failed to fetch available dates (${response.status})`, response.status);
      }

      return await response.json();
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError('Error contacting available-dates endpoint', 0, err);
    }
  },

  /**
   * Submits a real 15-depth vertical temperature reconstruction request.
   */
  async predict(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        let msg = `Prediction failed with status ${response.status}`;
        if (data && data.detail) {
          if (typeof data.detail === 'string') {
            msg = data.detail;
          } else if (typeof data.detail === 'object') {
            msg = data.detail.message || JSON.stringify(data.detail);
          }
        }
        throw new ApiError(msg, response.status, data);
      }

      return data as PredictionResponse;
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(err.message || 'Network error while contacting inference server', 0, err);
    }
  },
};
