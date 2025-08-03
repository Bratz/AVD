import { NextResponse } from 'next/server';
import crypto from 'crypto';
import { verificationStore } from '@/lib/auth/verification-store';
import { Resend } from 'resend';
import { brandingConfig } from '@/config/branding';

const resend = new Resend(process.env.RESEND_API_KEY);

export async function POST(request: Request) {
  try {
    const { email } = await request.json();
    
    if (!email || !email.includes('@')) {
      return NextResponse.json({ error: 'Invalid email' }, { status: 400 });
    }

    // Generate 6-digit code
    const code = crypto.randomInt(100000, 999999).toString();
    
    // Store code with 10-minute expiry
    verificationStore.set(email, code, 10);

    // Development mode - just log the code
    if (process.env.NODE_ENV === 'development' && !process.env.RESEND_API_KEY) {
      console.log(`[DEV MODE] Verification code for ${email}: ${code}`);
      return NextResponse.json({ 
        success: true, 
        dev_message: `Check console for code: ${code}` 
      });
    }

    // Production mode - send email
    try {
      await resend.emails.send({
        from: `${brandingConfig.appName} <noreply@${process.env.NEXT_PUBLIC_APP_DOMAIN || 'chatgbp.ai'}>`,
        to: email,
        subject: `Your ${brandingConfig.appName} verification code`,
        html: `
          <!DOCTYPE html>
          <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body style="margin: 0; padding: 0; background-color: #f4f4f4; font-family: Arial, sans-serif;">
              <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #000000; border-radius: 12px; padding: 40px; text-align: center;">
                  <!-- Logo -->
                  <div style="margin-bottom: 30px;">
                    <img src="${process.env.NEXT_PUBLIC_APP_URL || 'https://chatgbp.ai'}${brandingConfig.logos.main}" 
                         alt="${brandingConfig.appName}" 
                         style="height: 60px; width: auto;">
                  </div>
                  
                  <!-- Title -->
                  <h1 style="color: #ffffff; font-size: 24px; margin-bottom: 20px; font-weight: 600;">
                    Verification Code
                  </h1>
                  
                  <!-- Code Box -->
                  <div style="background-color: #1a1a1a; border-radius: 8px; padding: 24px; margin: 30px 0;">
                    <div style="color: #ffffff; font-size: 36px; letter-spacing: 8px; font-weight: bold; font-family: monospace;">
                      ${code}
                    </div>
                  </div>
                  
                  <!-- Instructions -->
                  <p style="color: #888888; font-size: 14px; line-height: 1.6; margin-bottom: 0;">
                    Enter this code on the sign-in page to continue.<br>
                    This code will expire in 10 minutes.
                  </p>
                  
                  <!-- Security Notice -->
                  <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #333333;">
                    <p style="color: #666666; font-size: 12px; margin: 0;">
                      If you didn't request this code, you can safely ignore this email.
                    </p>
                  </div>
                </div>
                
                <!-- Footer -->
                <div style="text-align: center; margin-top: 20px;">
                  <p style="color: #888888; font-size: 12px;">
                    © ${new Date().getFullYear()} ${brandingConfig.appName}. All rights reserved.
                  </p>
                </div>
              </div>
            </body>
          </html>
        `,
        text: `Your ${brandingConfig.appName} verification code is: ${code}\n\nThis code will expire in 10 minutes.`
      });

      console.log(`Verification email sent to ${email}`);
      return NextResponse.json({ success: true });
      
    } catch (emailError) {
      console.error('Failed to send email:', emailError);
      
      // In development, still return success but log the code
      if (process.env.NODE_ENV === 'development') {
        console.log(`[EMAIL FAILED] Verification code for ${email}: ${code}`);
        return NextResponse.json({ 
          success: true, 
          dev_message: `Email failed but code is: ${code}` 
        });
      }
      
      // In production, return error
      return NextResponse.json({ 
        error: 'Failed to send verification email' 
      }, { status: 500 });
    }

  } catch (error) {
    console.error('Verification error:', error);
    return NextResponse.json({ error: 'Failed to send code' }, { status: 500 });
  }
}