// utils/id-generator.ts
let counter = 0;

export function generateUniqueId(prefix: string = 'msg'): string {
  return `${prefix}-${Date.now()}-${counter++}-${Math.random().toString(36).substr(2, 9)}`;
}