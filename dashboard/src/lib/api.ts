/**
 * GlucoTwin - Clinical Dashboard API Client
 * Connects to the FastAPI backend at http://localhost:8000/api
 * Includes seamless fallback to mock data when backend is not running.
 * Amity University Mumbai - Team Aethan
 */

import {
  Patient,
  GlucoseReading,
  SpikePrediction,
  PatientAlert,
  DashboardData,
  InterventionSimulation,
  InterventionType
} from './types';
import {
  MOCK_PATIENTS,
  getMockDashboardData,
  generate24HourReadings,
  generatePrediction,
  generateAlerts,
  simulateIntervention as mockSimulateIntervention
} from './mockData';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
const TIMEOUT_MS = 2500; // Fast fallback if server isn't running

/**
 * Fetch wrapper with timeout
 */
async function fetchWithTimeout(url: string, options: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...(options.headers || {}),
      },
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Get list of all monitored T2D patients
 */
export async function getPatients(): Promise<Patient[]> {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/patients`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[GlucoTwin API] getPatients fell back to mock data:', err);
    return MOCK_PATIENTS;
  }
}

/**
 * Get individual patient profile
 */
export async function getPatient(id: string): Promise<Patient> {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/patients/${id}`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`[GlucoTwin API] getPatient(${id}) fell back to mock data:`, err);
    const found = MOCK_PATIENTS.find(p => p.id === id);
    if (!found) throw new Error(`Patient ${id} not found in mock records`);
    return found;
  }
}

/**
 * Get glucose & biometric readings for given hours
 */
export async function getReadings(id: string, hours: number = 24): Promise<GlucoseReading[]> {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/patients/${id}/readings?hours=${hours}`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`[GlucoTwin API] getReadings(${id}) fell back to mock data:`, err);
    return generate24HourReadings(id);
  }
}

/**
 * Get 2-hour spike prediction & trajectory
 */
export async function getPrediction(id: string): Promise<SpikePrediction> {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/patients/${id}/prediction`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`[GlucoTwin API] getPrediction(${id}) fell back to mock data:`, err);
    return generatePrediction(id);
  }
}

/**
 * Get active clinical alerts for a patient
 */
export async function getAlerts(id: string): Promise<PatientAlert[]> {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/patients/${id}/alerts`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`[GlucoTwin API] getAlerts(${id}) fell back to mock data:`, err);
    return generateAlerts(id);
  }
}

/**
 * Get consolidated dashboard payload for a patient
 */
export async function getDashboard(id: string): Promise<DashboardData> {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/patients/${id}/dashboard`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`[GlucoTwin API] getDashboard(${id}) fell back to mock data:`, err);
    return getMockDashboardData(id);
  }
}

/**
 * Run 'What-if' intervention simulation on the Digital Twin
 */
export async function simulateIntervention(
  id: string, 
  interventionType: InterventionType
): Promise<InterventionSimulation> {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/patients/${id}/simulate`, {
      method: 'POST',
      body: JSON.stringify({ intervention: interventionType }),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`[GlucoTwin API] simulateIntervention(${id}, ${interventionType}) fell back to mock simulation:`, err);
    return mockSimulateIntervention(id, interventionType);
  }
}
