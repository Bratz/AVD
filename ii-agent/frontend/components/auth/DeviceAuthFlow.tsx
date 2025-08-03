
// Simplified DeviceAuthFlow.tsx
import { useState, useEffect } from 'react';
import FingerprintJS from '@fingerprintjs/fingerprintjs-pro';

export function DeviceAuthFlow() {
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    checkDevice();
  }, []);
  
  const checkDevice = async () => {
    try {
      const fp = await FingerprintJS.load({
        apiKey: process.env.NEXT_PUBLIC_FINGERPRINT_API_KEY!
      });
      const result = await fp.get();
      
      // Check if device is known
      const response = await fetch('/api/auth/device-check', {
        method: 'POST',
        body: JSON.stringify({ visitorId: result.visitorId }),
        headers: { 'Content-Type': 'application/json' }
      });
      
      const { isKnown } = await response.json();
      
      if (isKnown) {
        // Auto-proceed for known devices
        window.location.href = '/';
      } else {
        // Show simple auth form
        setLoading(false);
      }
    } catch (error) {
      console.error('Device check failed:', error);
      setLoading(false);
    }
  };
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
      </div>
    );
  }
  
  // Simple email verification for new devices
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="bg-gray-900 p-8 rounded-lg">
        <h2 className="text-2xl font-bold mb-4">Welcome to chatGBP</h2>
        <p className="mb-4">Enter your email to continue</p>
        {/* Simple email form */}
      </div>
    </div>
  );
}