"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Hand, BookOpenText, Menu, X, Gamepad2, MessageSquareText } from "lucide-react";
import ThemeToggle from "./ThemeToggle";

const NAV_ITEMS = [
  { href: "/", label: "Susun Kata", icon: Hand },
  { href: "/tebak-kata", label: "Tebak Kata", icon: Gamepad2 },
  { href: "/uji-kalimat", label: "Uji Kalimat", icon: MessageSquareText },
  { href: "/kosakata", label: "Kosakata", icon: BookOpenText },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const NavLinks = () => (
    <ul className="flex flex-col gap-1">
      {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
        const active = pathname === href;
        return (
          <li key={href}>
            <Link
              href={href}
              onClick={() => setMobileOpen(false)}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                active
                  ? "bg-teal text-white"
                  : "text-ink-soft hover:bg-teal-soft hover:text-teal"
              }`}
            >
              <Icon className="h-[18px] w-[18px] shrink-0" />
              <span>{label}</span>
            </Link>
          </li>
        );
      })}
    </ul>
  );

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden md:flex md:w-60 md:flex-col md:shrink-0 border-r border-surface-border bg-surface px-4 py-6">
        <div className="flex items-center gap-2 px-2 pb-8">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal text-white">
            <Hand className="h-5 w-5" />
          </div>
          <div className="font-display text-lg font-semibold leading-none text-ink">
            SIBI
            <span className="block text-[11px] font-sans font-normal text-ink-soft mt-0.5">
              Recognition
            </span>
          </div>
        </div>

        <nav className="flex-1">
          <NavLinks />
        </nav>

        <div className="pt-4 border-t border-surface-border flex items-center justify-between px-2">
          <span className="text-xs text-ink-soft">Tampilan</span>
          <ThemeToggle />
        </div>
      </aside>

      {/* Mobile top bar */}
      <div className="md:hidden flex items-center justify-between border-b border-surface-border bg-surface px-4 py-3 sticky top-0 z-40">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal text-white">
            <Hand className="h-4 w-4" />
          </div>
          <span className="font-display text-base font-semibold text-ink">
            SIBI Recognition
          </span>
        </div>
        <button
          onClick={() => setMobileOpen(true)}
          aria-label="Buka menu"
          className="p-2 rounded-lg hover:bg-teal-soft text-ink"
        >
          <Menu className="h-5 w-5" />
        </button>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setMobileOpen(false)}
          />
          <div className="absolute left-0 top-0 h-full w-64 bg-surface p-5 flex flex-col gap-6 shadow-panel">
            <div className="flex items-center justify-between">
              <span className="font-display text-base font-semibold text-ink">
                SIBI Recognition
              </span>
              <button
                onClick={() => setMobileOpen(false)}
                aria-label="Tutup menu"
                className="p-1.5 rounded-lg hover:bg-teal-soft text-ink"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <NavLinks />
            <div className="mt-auto pt-4 border-t border-surface-border flex items-center justify-between">
              <span className="text-xs text-ink-soft">Tampilan</span>
              <ThemeToggle />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
