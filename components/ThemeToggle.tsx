"use client";

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => setMounted(true), []);

  const toggle = () => setTheme(theme === "dark" ? "light" : "dark");

  return (
    <button
      onClick={toggle}
      aria-label="Ganti tema"
      className="relative flex h-8 w-8 items-center justify-center rounded-lg text-ink-soft hover:bg-teal-soft hover:text-teal transition-colors"
    >
      {mounted && theme === "dark" ? (
        <Sun className="h-[18px] w-[18px]" />
      ) : (
        <Moon className="h-[18px] w-[18px]" />
      )}
    </button>
  );
}
