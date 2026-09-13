import React, { useRef, useCallback } from 'react';
import { MapPin, Navigation, Crosshair, Check } from 'lucide-react';

interface OceanMapProps {
  latitude: number;
  longitude: number;
  onSelectCoordinates: (lat: number, lon: number) => void;
  disabled?: boolean;
}

// Bounding box of North Indian Ocean model domain
const LAT_MIN = 5.0;
const LAT_MAX = 30.0;
const LON_MIN = 45.0;
const LON_MAX = 105.0;

// High-profile demonstration presets
const PRESET_LOCATIONS = [
  { name: 'Bay of Bengal (Default)', lat: 15.0, lon: 85.0 },
  { name: 'Central Arabian Sea', lat: 14.0, lon: 65.0 },
  { name: 'Southern NIO / Sri Lanka', lat: 8.0, lon: 88.0 },
  { name: 'Northern Arabian Sea', lat: 20.0, lon: 68.0 },
  { name: 'Andaman Sea Basin', lat: 10.0, lon: 94.0 },
];

export const OceanMap: React.FC<OceanMapProps> = ({
  latitude,
  longitude,
  onSelectCoordinates,
  disabled = false,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  // Convert (lon, lat) to SVG coordinate percentages
  // X: 45°E -> 0%, 105°E -> 100%
  // Y: 30°N -> 0%, 5°N -> 100%
  const xPercent = Math.max(0, Math.min(100, ((longitude - LON_MIN) / (LON_MAX - LON_MIN)) * 100));
  const yPercent = Math.max(0, Math.min(100, ((LAT_MAX - latitude) / (LAT_MAX - LAT_MIN)) * 100));

  // Compute mapped 0.25° canonical grid coordinates
  const latIdx = Math.max(0, Math.min(100, Math.round((latitude - LAT_MIN) / 0.25)));
  const lonIdx = Math.max(0, Math.min(240, Math.round((longitude - LON_MIN) / 0.25)));
  const gridLat = (LAT_MIN + latIdx * 0.25).toFixed(2);
  const gridLon = (LON_MIN + lonIdx * 0.25).toFixed(2);

  // Handle click on map SVG
  const handleMapClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (disabled) return;
      if (!svgRef.current) return;

      const rect = svgRef.current.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const rawLon = LON_MIN + (clickX / rect.width) * (LON_MAX - LON_MIN);
      const rawLat = LAT_MAX - (clickY / rect.height) * (LAT_MAX - LAT_MIN);

      // Snap to 0.25° grid step or round to 2 decimals
      const roundedLat = Math.round(Math.max(LAT_MIN, Math.min(LAT_MAX, rawLat)) * 100) / 100;
      const roundedLon = Math.round(Math.max(LON_MIN, Math.min(LON_MAX, rawLon)) * 100) / 100;

      onSelectCoordinates(roundedLat, roundedLon);
    },
    [disabled, onSelectCoordinates]
  );

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <Navigation className="w-4 h-4 text-sky-700" />
            North Indian Ocean Domain
          </h2>
          <p className="text-xs text-slate-500">
            Click to set coordinate or choose a preset basin location
          </p>
        </div>

        {/* Selected Coordinates Readout Badge */}
        <div className="hidden sm:flex items-center gap-2 text-xs font-mono bg-slate-100 border border-slate-200 px-2.5 py-1 rounded">
          <Crosshair className="w-3.5 h-3.5 text-slate-500" />
          <span className="font-semibold text-slate-800">{latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E</span>
          <span className="text-slate-400">→</span>
          <span className="text-slate-600">Grid: {gridLat}°N, {gridLon}°E</span>
        </div>
      </div>

      {/* Preset Location Quick-Select Chips */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {PRESET_LOCATIONS.map((preset) => {
          const isSelected =
            Math.abs(latitude - preset.lat) < 0.1 && Math.abs(longitude - preset.lon) < 0.1;
          return (
            <button
              key={preset.name}
              type="button"
              disabled={disabled}
              onClick={() => onSelectCoordinates(preset.lat, preset.lon)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors border ${
                isSelected
                  ? 'bg-sky-50 text-sky-800 border-sky-300 font-semibold'
                  : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              {isSelected && <Check className="w-3 h-3 inline mr-1 text-sky-600" />}
              {preset.name}
            </button>
          );
        })}
      </div>

      {/* Interactive SVG Ocean Map */}
      <div className="relative border border-slate-300 rounded-lg overflow-hidden select-none bg-[#e8f1f5]">
        <svg
          ref={svgRef}
          viewBox="0 0 600 350"
          className="w-full h-auto cursor-crosshair block"
          onClick={handleMapClick}
        >
          {/* Water Background */}
          <rect width="600" height="350" fill="#e2edf2" />

          {/* Bathymetry / Ocean Depth Subtle Contours */}
          <ellipse cx="200" cy="180" rx="140" ry="90" fill="#d7e6ec" opacity="0.6" />
          <ellipse cx="400" cy="190" rx="110" ry="80" fill="#d7e6ec" opacity="0.6" />

          {/* Coordinate Graticules (5° Lat, 10° Lon intervals) */}
          {/* Longitudes: 50°, 60°, 70°, 80°, 90°, 100° */}
          {[50, 60, 70, 80, 90, 100].map((lon) => {
            const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * 600;
            return (
              <g key={`lon-${lon}`}>
                <line x1={x} y1="0" x2={x} y2="350" stroke="#cbd5e1" strokeWidth="0.8" strokeDasharray="3,3" />
                <text x={x + 3} y="342" fill="#64748b" fontSize="10" fontFamily="monospace">
                  {lon}°E
                </text>
              </g>
            );
          })}

          {/* Latitudes: 10°, 15°, 20°, 25° */}
          {[10, 15, 20, 25].map((lat) => {
            const y = ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * 350;
            return (
              <g key={`lat-${lat}`}>
                <line x1="0" y1={y} x2="600" y2={y} stroke="#cbd5e1" strokeWidth="0.8" strokeDasharray="3,3" />
                <text x="6" y={y - 4} fill="#64748b" fontSize="10" fontFamily="monospace">
                  {lat}°N
                </text>
              </g>
            );
          })}

          {/* Geographic Landmass Outlines (Light Olive / Tan Landmass Styling) */}
          {/* 1. Arabian Peninsula & Middle East */}
          <path
            d="M 0,0 L 150,0 L 140,50 L 120,80 L 100,100 L 115,130 L 105,150 L 80,165 L 50,180 L 0,195 Z"
            fill="#dbe2d8"
            stroke="#94a3b8"
            strokeWidth="1"
          />
          {/* Persian Gulf / Strait of Hormuz water notch */}
          <path
            d="M 110,40 Q 130,55 145,50 L 135,20 Z"
            fill="#e2edf2"
            stroke="#94a3b8"
            strokeWidth="0.7"
          />

          {/* 2. Indian Subcontinent & Pakistan/Iran Coast */}
          <path
            d="M 140,0 L 320,0 L 330,40 L 350,80 L 320,110 L 305,150 L 290,200 L 275,250 L 260,265 L 245,245 L 240,210 L 225,170 L 205,135 L 180,120 L 160,110 L 140,90 Z"
            fill="#dbe2d8"
            stroke="#94a3b8"
            strokeWidth="1.2"
          />

          {/* 3. Sri Lanka */}
          <ellipse
            cx="278"
            cy="275"
            rx="12"
            ry="18"
            fill="#dbe2d8"
            stroke="#94a3b8"
            strokeWidth="1"
          />

          {/* 4. Southeast Asia / Myanmar / Thailand / Malacca / Sumatra */}
          <path
            d="M 430,0 L 600,0 L 600,350 L 560,350 L 530,300 L 520,260 L 515,220 L 500,180 L 485,150 L 460,120 L 435,90 L 415,60 L 410,20 Z"
            fill="#dbe2d8"
            stroke="#94a3b8"
            strokeWidth="1.2"
          />

          {/* 5. Andaman & Nicobar Islands (Chain) */}
          <g fill="#94a3b8">
            <circle cx="480" cy="180" r="2.5" />
            <circle cx="482" cy="195" r="2.5" />
            <circle cx="483" cy="210" r="2.5" />
            <circle cx="485" cy="235" r="2.5" />
            <circle cx="487" cy="250" r="2.5" />
          </g>

          {/* 6. Maldives Chain (Subtle at bottom left) */}
          <g fill="#94a3b8" opacity="0.7">
            <circle cx="240" cy="300" r="1.5" />
            <circle cx="241" cy="315" r="1.5" />
            <circle cx="242" cy="330" r="1.5" />
          </g>

          {/* Major Basin Labels (Subtle Cartographic Text) */}
          <text x="140" y="210" fill="#334155" fontSize="12" fontWeight="bold" letterSpacing="2" opacity="0.75">
            ARABIAN SEA
          </text>
          <text x="360" y="200" fill="#334155" fontSize="12" fontWeight="bold" letterSpacing="2" opacity="0.75">
            BAY OF BENGAL
          </text>
          <text x="240" y="110" fill="#475569" fontSize="11" fontWeight="600" opacity="0.8">
            INDIA
          </text>
          <text x="285" y="325" fill="#334155" fontSize="10" letterSpacing="1" opacity="0.6">
            EQUATORIAL INDIAN OCEAN
          </text>
          <text x="495" y="170" fill="#334155" fontSize="9" opacity="0.65">
            ANDAMAN SEA
          </text>

          {/* Domain Outer Boundary */}
          <rect x="1" y="1" width="598" height="348" fill="none" stroke="#475569" strokeWidth="1.5" />

          {/* Target Reticle / Location Marker */}
          <g transform={`translate(${(xPercent / 100) * 600}, ${(yPercent / 100) * 350})`}>
            {/* Outer animated ping ring */}
            <circle cx="0" cy="0" r="14" fill="none" stroke="#0284c7" strokeWidth="1.5" strokeDasharray="3,2" opacity="0.85" />
            {/* Crosshair lines */}
            <line x1="-18" y1="0" x2="-6" y2="0" stroke="#0369a1" strokeWidth="2" />
            <line x1="6" y1="0" x2="18" y2="0" stroke="#0369a1" strokeWidth="2" />
            <line x1="0" y1="-18" x2="0" y2="-6" stroke="#0369a1" strokeWidth="2" />
            <line x1="0" y1="6" x2="0" y2="18" stroke="#0369a1" strokeWidth="2" />
            {/* Center target dot */}
            <circle cx="0" cy="0" r="4" fill="#0284c7" stroke="#ffffff" strokeWidth="2" />
          </g>
        </svg>

        {/* Small Domain Legend in Corner */}
        <div className="absolute bottom-2 left-2 bg-white/90 backdrop-blur-xs border border-slate-200 px-2 py-1 rounded text-[10px] text-slate-600 font-mono pointer-events-none">
          Domain: 5°N–30°N | 45°E–105°E
        </div>
      </div>
    </div>
  );
};
