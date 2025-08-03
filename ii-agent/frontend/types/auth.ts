// types/auth.ts
export interface User {
  id: string;
  email: string;
  name?: string;
  role: string;
  createdAt: Date;
}

export interface DeviceInfo {
  id: string;
  deviceName: string;
  deviceType: 'desktop' | 'mobile' | 'tablet';
  location: string;
  lastSeen: string;
  trustScore: number;
  userName: string;
  os: string;
  browser: string;
}

export interface AuthContext {
  ip?: string;
  userAgent?: string;
  timestamp: number;
}

export interface Anomaly {
  type: 'impossible_travel' | 'hardware_change' | 'behavioral_anomaly';
  severity: 'low' | 'medium' | 'high';
  details: Record<string, any>;
}

export interface AnomalyReport {
  anomalies: Anomaly[];
  riskScore: number;
  requiresMFA: boolean;
  recommendations: string[];
}

export interface AuthResult {
  level: 'new_device' | 'trusted' | 'recognized' | 'suspicious' | 'blocked';
  user?: User;
  requiresAction: 'none' | 'quick_verify' | 'mfa' | 'full_auth';
}

export interface Session {
  id: string;
  userId: string;
  deviceId: string;
  createdAt: Date;
  expiresAt: Date;
}

export interface AuthLog {
  id: string;
  deviceId: string;
  action: string;
  timestamp: Date;
  success: boolean;
}

export enum DeviceType {
  DESKTOP = 'desktop',
  MOBILE = 'mobile',
  TABLET = 'tablet'
}