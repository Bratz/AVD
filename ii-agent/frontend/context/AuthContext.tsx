"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import Cookies from 'js-cookie';

interface User {
  id: string;
  email: string;
  name?: string;
  role?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // In development with skip auth, auto-set user
  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_SKIP_AUTH === 'true') {
      setUser({
        id: 'dev-user-123',
        email: 'dev@example.com',
        name: 'Dev User',
        role: 'admin'
      });
      setIsLoading(false);
      return;
    }

    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const token = Cookies.get('auth-token');
      
      if (!token) {
        setUser(null);
        return;
      }

      // In development, accept dev token
      if (process.env.NODE_ENV === 'development' && token === 'dev-token-123') {
        setUser({
          id: 'dev-user-123',
          email: 'dev@example.com',
          name: 'Dev User',
          role: 'admin'
        });
        return;
      }

      // Normal auth check - call your API
      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        setUser(null);
        Cookies.remove('auth-token');
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    // Your login implementation
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });

    if (response.ok) {
      const data = await response.json();
      setUser(data.user);
      router.push('/');
    } else {
      throw new Error('Login failed');
    }
  };

  const logout = async () => {
    Cookies.remove('auth-token');
    setUser(null);
    router.push('/auth');
  };

  return (
    <AuthContext.Provider value={{ 
      user, 
      isAuthenticated: !!user,
      isLoading,
      login, 
      logout, 
      checkAuth 
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};