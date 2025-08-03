"use client";

export function DevLogin() {
  // Only show in development
  if (process.env.NODE_ENV !== 'development') return null;

  const handleDevLogin = async () => {
    const response = await fetch('/api/auth/dev-login', {
      method: 'POST',
    });
    
    if (response.ok) {
      window.location.href = '/';
    }
  };

  return (
    <button
      onClick={handleDevLogin}
      className="fixed bottom-4 right-4 bg-yellow-500 text-black px-4 py-2 rounded-lg 
                 font-bold shadow-lg hover:bg-yellow-400 transition-colors z-50"
    >
      Dev Login
    </button>
  );
}