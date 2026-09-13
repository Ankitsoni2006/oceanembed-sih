import React, { useState, useEffect, useCallback } from 'react';
import { api } from './services/api';
import {
  HealthResponse,
  AvailableDatesResponse,
  PredictionResponse,
} from './types';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { OceanMap } from './components/OceanMap';
import { ControlPanel } from './components/ControlPanel';
import { ProfileChart } from './components/ProfileChart';
import { DepthInspector } from './components/DepthInspector';
import { ProfileMetrics } from './components/ProfileMetrics';
import { SurfaceObservationsPanel } from './components/SurfaceObservationsPanel';
import { ModelPerformance } from './components/ModelPerformance';
import { Methodology } from './components/Methodology';
import { Limitations } from './components/Limitations';
import { Footer } from './components/Footer';

export default function App() {
  // State: Coordinates & Date (Initial default: Central Bay of Bengal, benchmark date)
  const [latitude, setLatitude] = useState<number>(15.0);
  const [longitude, setLongitude] = useState<number>(85.0);
  const [date, setDate] = useState<string>('2020-09-15');

  // State: Backend Connectivity & Available Dates
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHealthLoading, setIsHealthLoading] = useState<boolean>(true);
  const [availableDates, setAvailableDates] = useState<AvailableDatesResponse | null>(null);
  const [isDatesLoading, setIsDatesLoading] = useState<boolean>(true);

  // State: Real Model Prediction & Interactive Depth Selection
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [selectedDepth, setSelectedDepth] = useState<number | null>(100);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Health check polling
  const checkBackendHealth = useCallback(async () => {
    try {
      const data = await api.checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setIsHealthLoading(false);
    }
  }, []);

  // Fetch available dates archive
  const fetchDates = useCallback(async () => {
    setIsDatesLoading(true);
    try {
      const data = await api.getAvailableDates();
      setAvailableDates(data);
      if (data.dates.length > 0 && !data.dates.includes('2020-09-15')) {
        setDate(data.dates[0]);
      }
    } catch (err: any) {
      console.warn('Failed to load available dates from backend:', err);
    } finally {
      setIsDatesLoading(false);
    }
  }, []);

  // Execute inference request against real FastAPI backend
  const handleReconstruct = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await api.predict({
        latitude,
        longitude,
        date,
      });
      setPrediction(response);
      if (selectedDepth === null) {
        setSelectedDepth(100);
      }
    } catch (err: any) {
      // Clear stale prediction on error so old profile is not misleadingly displayed
      setPrediction(null);
      setError(err.message || 'Inference error occurred while contacting the server.');
    } finally {
      setIsLoading(false);
    }
  }, [latitude, longitude, date, selectedDepth]);

  // Initial mount: check health and get available dates (no auto-prediction to allow clean jury demonstration)
  useEffect(() => {
    checkBackendHealth();
    fetchDates();

    // Poll health status every 15 seconds
    const interval = setInterval(checkBackendHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Handlers for coordinate updates from map or control panel
  const handleSelectCoordinates = useCallback((lat: number, lon: number) => {
    setLatitude(lat);
    setLongitude(lon);
    setError(null);
  }, []);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 flex flex-col font-sans antialiased selection:bg-sky-500 selection:text-white">
      {/* 1. Header with live FastAPI status */}
      <Header health={health} isHealthLoading={isHealthLoading} />

      {/* 2. Hero Section */}
      <Hero />

      {/* 3. Main Workspace Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Workspace Grid: Controls & Map (Left) vs Profile & Diagnostics (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* Left Column: Controls and Map (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            <ControlPanel
              latitude={latitude}
              longitude={longitude}
              onChangeLatitude={(lat) => {
                setError(null);
                setLatitude(lat);
              }}
              onChangeLongitude={(lon) => {
                setError(null);
                setLongitude(lon);
              }}
              date={date}
              onChangeDate={(dt) => {
                setError(null);
                setDate(dt);
              }}
              availableDates={availableDates}
              isDatesLoading={isDatesLoading}
              onReconstruct={handleReconstruct}
              isLoading={isLoading}
              error={error}
              onClearError={() => setError(null)}
            />

            <OceanMap
              latitude={latitude}
              longitude={longitude}
              onSelectCoordinates={handleSelectCoordinates}
              disabled={isLoading}
            />
          </div>

          {/* Right Column: Reconstructed Profile & Oceanographic Metrics (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            <ProfileChart
              prediction={prediction}
              selectedDepth={selectedDepth}
              onSelectDepth={(d) => setSelectedDepth(d)}
              isLoading={isLoading}
            />

            {prediction && (
              <>
                <DepthInspector
                  depths={prediction.depths_m}
                  temperatures={prediction.temperatures_c}
                  selectedDepth={selectedDepth}
                  onSelectDepth={(d) => setSelectedDepth(d)}
                />

                <ProfileMetrics prediction={prediction} />

                <SurfaceObservationsPanel
                  observations={prediction.surface_observations}
                  date={prediction.date}
                />
              </>
            )}
          </div>
        </div>

        {/* Section 4: Validation Results */}
        <ModelPerformance />

        {/* Section 5: Architecture & Reconstruction Pipeline */}
        <Methodology />

        {/* Section 6: Scope & Limitations */}
        <Limitations />
      </main>

      {/* 7. Footer */}
      <Footer />
    </div>
  );
}
