// context/DevAuthContext.tsx
import { ThemeProvider } from '@/components/theme-provider';
import { createContext, useContext } from 'react';
import { AuthProvider } from './AuthContext';

const DevAuthContext = createContext({
  user: {
    id: 'dev-user-123',
    email: 'dev@example.com',
    name: 'Dev User',
    role: 'admin',
  },
  isAuthenticated: true,
  login: async () => {},
  logout: async () => {},
});

export function DevAuthProvider({ children }: { children: React.ReactNode }) {
  return (
    <DevAuthContext.Provider value={{ 
      user: {
        id: 'dev-user-123',
        email: 'dev@example.com',
        name: 'Dev User',
        role: 'admin',
      },
      isAuthenticated: true,
      login: async () => {},
      logout: async () => {},
    }}>
      {children}
    </DevAuthContext.Provider>
  );
}

// In your main provider:
export default function Providers({ children }: { children: React.ReactNode }) {
  const isDev = process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_SKIP_AUTH === 'true';
  
  return (
    <ThemeProvider>
      {isDev ? (
        <DevAuthProvider>{children}</DevAuthProvider>
      ) : (
        <AuthProvider>{children}</AuthProvider>
      )}
    </ThemeProvider>
  );
}