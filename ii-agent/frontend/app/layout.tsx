import type { Metadata } from "next";
import "./globals.css";
import Providers from "@/providers";
import { getAppTitle, brandingConfig } from "@/config/branding";

export const metadata: Metadata = {
  title: getAppTitle(),
  description: brandingConfig.appDescription,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href={`${brandingConfig.logos.favicon}/apple-touch-icon.png`}
        />
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href={`${brandingConfig.logos.favicon}/favicon-32x32.png`}
        />
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href={`${brandingConfig.logos.favicon}/favicon-16x16.png`}
        />
        <link rel="manifest" href={`${brandingConfig.logos.favicon}/site.webmanifest`} />
      </head>
      <body className={`antialiased`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}