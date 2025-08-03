// Singleton pattern for verification codes
class VerificationStore {
  private static instance: VerificationStore;
  private codes: Map<string, { code: string; expires: number }>;
  
  private constructor() {
    this.codes = new Map();
    // Clean up expired codes every minute
    setInterval(() => this.cleanup(), 60000);
  }
  
  static getInstance(): VerificationStore {
    if (!VerificationStore.instance) {
      VerificationStore.instance = new VerificationStore();
    }
    return VerificationStore.instance;
  }
  
  set(email: string, code: string, expiresInMinutes: number = 10): void {
    this.codes.set(email, {
      code,
      expires: Date.now() + expiresInMinutes * 60 * 1000
    });
  }
  
  get(email: string): { code: string; expires: number } | undefined {
    return this.codes.get(email);
  }
  
  delete(email: string): void {
    this.codes.delete(email);
  }
  
  private cleanup(): void {
    const now = Date.now();
    for (const [email, data] of this.codes.entries()) {
      if (data.expires < now) {
        this.codes.delete(email);
      }
    }
  }
}

export const verificationStore = VerificationStore.getInstance();