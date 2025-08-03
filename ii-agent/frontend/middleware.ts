// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { jwtVerify } from 'jose';

const secret = new TextEncoder().encode(
  process.env.JWT_SECRET || 'your-secret-key-min-32-chars-long!'
);

export async function middleware(request: NextRequest) {
    if (process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_SKIP_AUTH === 'true') {
    return NextResponse.next();
    }

  // Note: In middleware, cookies are accessed differently
  const token = request.cookies.get('auth-token');
  
  // Allow public routes
  if (request.nextUrl.pathname.startsWith('/auth') || 
      request.nextUrl.pathname.startsWith('/api/auth')) {
    return NextResponse.next();
  }
  
  if (!token) {
    return NextResponse.redirect(new URL('/auth', request.url));
  }
  
  try {
    await jwtVerify(token.value, secret);
    return NextResponse.next();
  } catch {
    // Invalid token, redirect to auth
    const response = NextResponse.redirect(new URL('/auth', request.url));
    response.cookies.delete('auth-token');
    return response;
  }
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)']
};