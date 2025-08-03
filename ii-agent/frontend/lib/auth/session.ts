// lib/auth/session.ts
import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';

export interface UserSession {
  id: string;
  email: string;
  role: string;
  features: string[];
}

export const createSession = async (user: UserSession) => {
  const token = jwt.sign(
    {
      ...user,
      iat: Date.now(),
      exp: Date.now() + 7 * 24 * 60 * 60 * 1000, // 7 days
    },
    process.env.JWT_SECRET!
  );
  
  // Add await here
  const cookieStore = await cookies();
  cookieStore.set('auth-token', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 7 * 24 * 60 * 60,
  });
  
  return token;
};