import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { SignJWT } from 'jose';
import { verificationStore } from '@/lib/auth/verification-store';

// HARDCODED VERIFICATION CODE - TEMPORARY
const HARDCODED_VERIFICATION_CODE = '111111';

const secret = new TextEncoder().encode(
  process.env.JWT_SECRET || 'your-secret-key-min-32-chars-long!'
);

export async function POST(request: Request) {
  try {
    const { email, code } = await request.json();
    
    // Get stored code
    const stored = verificationStore.get(email);
    
    if (!stored || stored.expires < Date.now()) {
      return NextResponse.json({ error: 'Code expired' }, { status: 400 });
    }
    
    if (stored.code !== code) {
      return NextResponse.json({ error: 'Invalid code' }, { status: 400 });
    }
    
    // Clear used code
    verificationStore.delete(email);
    
    if (code !== HARDCODED_VERIFICATION_CODE) {
      return NextResponse.json(
        { 
          error: 'Invalid verification code',
          message: 'Please enter the correct verification code'
        },
        { status: 401 }
      );
    }
    // Create session token using jose
    const token = await new SignJWT({ email })
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime('7d')
      .sign(secret);
    
    // Get cookies instance with await (Next.js 15)
    const cookieStore = await cookies();
    
    // Set cookie
    cookieStore.set('auth-token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 7 * 24 * 60 * 60 // 7 days
    });
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Verification error:', error);
    return NextResponse.json({ error: 'Verification failed' }, { status: 500 });
  }
}