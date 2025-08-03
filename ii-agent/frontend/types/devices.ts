// types/device.ts
import { DeviceType, Session, AuthLog } from './auth';
export interface Device {
  id: string;
  userId: string;
  fingerprintHash: string;
  visitorId: string;
  hardware: any;
  browser: any;
  name: string;
  type: DeviceType;
  os: string;
  lastIp: string;
  lastLocation?: string;
  trustScore: number;
  isTrusted: boolean;
  isBlocked: boolean;
  behavioralProfile?: any;
  anomalyScore: number;
  firstSeen: Date;
  lastSeen: Date;
  lastAuth?: Date;
  sessions: Session[];
  authLogs: AuthLog[];
}