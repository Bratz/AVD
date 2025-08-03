// app/api/auth/dev-login/route.ts
import { NextResponse } from 'next/server';

export async function POST() {
  if (process.env.NODE_ENV !== 'development') {
    return NextResponse.json({ error: 'Not allowed' }, { status: 403 });
  }

  const response = NextResponse.json({ 
    success: true,
    user: {
      id: 'dev-user-123',
      email: 'dev@example.com',
      name: 'Dev User',
    }
  });

  // Set auth cookie
  response.cookies.set('auth-token', 'dev-token-123', {
    httpOnly: true,
    secure: false,
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7, // 7 days
  });

  return response;
}