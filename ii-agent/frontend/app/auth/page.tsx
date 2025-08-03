"use client";

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Mail, ArrowRight, Loader2 } from 'lucide-react';
import Image from 'next/image';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import { brandingConfig } from '@/config/branding';

export default function AuthPage() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [codeSent, setCodeSent] = useState(false);
  const [code, setCode] = useState('');
  const router = useRouter();

  useEffect(() => {
    // Auto-login in development
    if (process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_SKIP_AUTH === 'true') {
      // Set a dev token
      document.cookie = 'auth-token=dev-token; path=/';
      router.push('/');
    }
  }, [router]);

  const handleSendCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;

    setLoading(true);
    try {
      const response = await fetch('/api/auth/send-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });

      if (response.ok) {
        setCodeSent(true);
        toast.success('Verification code sent to your email');
      } else {
        toast.error('Failed to send code. Please try again.');
      }
    } catch {
      toast.error('Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code) return;

    setLoading(true);
    try {
      const response = await fetch('/api/auth/verify-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, code })
      });

      if (response.ok) {
        toast.success(`Welcome to ${brandingConfig.appName}!`);
        router.push('/');
      } else {
        toast.error('Invalid code. Please try again.');
      }
    } catch {
      toast.error('Verification failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <Image
            src={brandingConfig.logos.main}
            alt={brandingConfig.appName}
            width={60}
            height={60}
            className="mx-auto mb-4"
            priority
          />
          <h1 className="text-2xl font-bold text-foreground mb-2">
            Welcome to {brandingConfig.appName}
          </h1>
          <p className="text-muted-foreground">
            {codeSent ? 'Enter the code sent to your email' : 'Sign in or create an account'}
          </p>
        </div>

        {/* Form */}
        <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-8">
          {!codeSent ? (
            <form onSubmit={handleSendCode} className="space-y-4">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                  Email address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
                  <input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg 
                             text-foreground placeholder-gray-500 focus:outline-none focus:border-blue-500 
                             transition-colors"
                    placeholder="you@example.com"
                    required
                    disabled={loading}
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading || !email}
                className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 
                         disabled:cursor-not-allowed text-foreground font-medium rounded-lg 
                         transition-all duration-200 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <>
                    Continue
                    <ArrowRight className="h-5 w-5" />
                  </>
                )}
              </button>
            </form>
          ) : (
            <form onSubmit={handleVerifyCode} className="space-y-4">
              <div>
                <label htmlFor="code" className="block text-sm font-medium text-gray-300 mb-2">
                  Verification code
                </label>
                <input
                  id="code"
                  type="text"
                  value={code}
                  onChange={(e) => setCode(e.target.value.replace(/\s/g, ''))}
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg 
                           text-foreground text-center text-xl tracking-widest placeholder-gray-500 
                           focus:outline-none focus:border-blue-500 transition-colors"
                  placeholder="000000"
                  maxLength={6}
                  required
                  disabled={loading}
                  autoFocus
                />
              </div>

              <button
                type="submit"
                disabled={loading || code.length !== 6}
                className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 
                         disabled:cursor-not-allowed text-foreground font-medium rounded-lg 
                         transition-all duration-200 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  'Verify'
                )}
              </button>

              <button
                type="button"
                onClick={() => {
                  setCodeSent(false);
                  setCode('');
                }}
                className="w-full text-muted-foreground hover:text-foreground text-sm transition-colors"
              >
                Use a different email
              </button>
            </form>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-gray-500 text-sm mt-6">
          By continuing, you agree to our Terms of Service
        </p>
      </motion.div>
    </div>
  );
}