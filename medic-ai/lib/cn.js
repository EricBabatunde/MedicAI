import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges class names using clsx and tailwind-merge.
 * Usage: cn("px-4 py-2", isActive && "bg-brand-500", className)
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

