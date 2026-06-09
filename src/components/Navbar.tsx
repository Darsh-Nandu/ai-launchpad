import React, { useState } from "react";
import { HubType } from "../types";
import { Menu, X, Sparkles, BookOpen, Layers, Monitor, ChevronRight, Brain, Rocket } from "lucide-react";

interface NavbarProps {
  currentHub: HubType;
  onChangeHub: (hub: HubType) => void;
}

export default function Navbar({ currentHub, onChangeHub }: NavbarProps) {
  const [isOpen, setIsOpen] = useState(false);

  // Styling helpers based on current theme hub
  const getNavbarStyles = () => {
    switch (currentHub) {
      case HubType.PLAYGROUND:
        return {
          bg: "bg-[#ece9e6]/80 text-[#2c2b30] border-b-2 border-black/15",
          btnActive: "bg-[#ffe599] text-black border-2 border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]",
          btnInactive: "text-gray-700 hover:bg-black/5 rounded-lg",
          logo: "font-black tracking-tight",
        };
      case HubType.STUDIO:
        return {
          bg: "bg-[#f8fafc]/85 text-slate-800 border-b border-slate-200/80",
          btnActive: "bg-[#1a56db] text-white shadow-[0_4px_12px_rgba(26,86,219,0.22)] rounded-lg",
          btnInactive: "text-slate-600 hover:bg-slate-200/50 hover:text-slate-900 rounded-lg",
          logo: "font-extrabold tracking-wider text-[#1a56db]",
        };
      case HubType.LIBRARY:
        return {
          bg: "bg-[#faf7f2]/80 text-[#2c2b30] border-b border-[#eae3d9]",
          btnActive: "bg-[#e56845] text-white",
          btnInactive: "text-[#646268] hover:bg-[#eae3d9]/40 hover:text-black rounded-lg",
          logo: "font-serif italic font-semibold hover:text-[#e56845]",
        };
    }
  };

  const navStyles = getNavbarStyles();

  // Helper smooth scroll handler
  const handleScrollTo = (id: string) => {
    setIsOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <nav className={`fixed top-0 left-0 right-0 h-16 backdrop-blur-md z-50 transition-all duration-300 ${navStyles.bg}`}>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-full flex items-center justify-between">
        
        {/* LOGO */}
        <div
          onClick={() => handleScrollTo(currentHub === HubType.PLAYGROUND ? "retro-hero" : currentHub === HubType.STUDIO ? "studio-hub" : "library-hub")}
          className={`flex items-center gap-2 cursor-pointer text-base md:text-lg font-extrabold select-none ${navStyles.logo}`}
        >
          <div className="flex items-center gap-1.5 shrink-0">
            <Brain className="w-5 h-5 text-indigo-500 animate-pulse stroke-[2.5]" />
            <Rocket className="w-4 h-4 text-[#e56845] stroke-[2.5] transform rotate-45" />
          </div>
          <span>AI Launchpad</span>
        </div>

        {/* HUB TOGGLERS (CHAMBERS) - DESKTOP */}
        <div className="hidden md:flex items-center gap-2">
          <button
            onClick={() => onChangeHub(HubType.PLAYGROUND)}
            className={`px-3 py-1.5 text-xs uppercase tracking-wider font-extrabold flex items-center gap-1.5 cursor-pointer transition-all ${
              currentHub === HubType.PLAYGROUND ? navStyles.btnActive : navStyles.btnInactive
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Playground</span>
          </button>

          <button
            onClick={() => onChangeHub(HubType.STUDIO)}
            className={`px-3 py-1.5 text-xs uppercase tracking-wider font-extrabold flex items-center gap-1.5 cursor-pointer transition-all ${
              currentHub === HubType.STUDIO ? navStyles.btnActive : navStyles.btnInactive
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Roadmaps & Projects</span>
          </button>

          <button
            onClick={() => onChangeHub(HubType.LIBRARY)}
            className={`px-3 py-1.5 text-xs uppercase tracking-wider font-extrabold flex items-center gap-1.5 cursor-pointer transition-all ${
              currentHub === HubType.LIBRARY ? navStyles.btnActive : navStyles.btnInactive
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Resources & Videos</span>
          </button>
        </div>

        {/* CORE HUB INTERNAL NAVIGATION LINKS */}
        <div className="hidden md:flex items-center gap-4 text-xs font-bold uppercase tracking-wider">
          {currentHub === HubType.PLAYGROUND && (
            <>
              <button onClick={() => handleScrollTo("retro-hero")} className="hover:text-amber-500 transition-colors cursor-pointer">
                Home
              </button>
              <button onClick={() => handleScrollTo("ai-doubt-solver-section")} className="hover:text-amber-500 transition-colors cursor-pointer">
                Doubt Solver
              </button>
            </>
          )}

          {currentHub === HubType.STUDIO && (
            <>
              <button onClick={() => handleScrollTo("studio-hub")} className="hover:text-[#1a56db] transition-colors cursor-pointer">
                Syllabus
              </button>
              <button onClick={() => handleScrollTo("projects-section")} className="hover:text-[#1a56db] transition-colors cursor-pointer">
                Projects
              </button>
            </>
          )}

          {currentHub === HubType.LIBRARY && (
            <>
              <button onClick={() => handleScrollTo("library-hub")} className="hover:text-[#e56845] transition-colors cursor-pointer">
                Library
              </button>
              <button onClick={() => handleScrollTo("videos-section")} className="hover:text-[#e56845] transition-colors cursor-pointer">
                Videos
              </button>
            </>
          )}
        </div>

        {/* MOBILE MENU TOGGLE */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="md:hidden p-2 rounded-lg hover:bg-black/5 text-[#2c2b30] flex items-center justify-center cursor-pointer"
        >
          {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>

      </div>

      {/* MOBILE DRAWER */}
      {isOpen && (
        <div className={`absolute top-16 left-0 right-0 py-6 px-4 md:hidden flex flex-col gap-5 border-b shadow-2xl transition-all ${
          currentHub === HubType.PLAYGROUND
            ? "bg-[#ece9e6] border-black/15 text-[#2c2b30]"
            : currentHub === HubType.STUDIO
            ? "bg-[#f8fafc] border-slate-200 text-slate-800"
            : "bg-[#faf7f2] border-[#eae3d9] text-[#2c2b30]"
        }`}>
          <div>
            <span className="text-[10px] font-mono font-extrabold uppercase text-gray-500 tracking-wider">Chambers Switcher</span>
            <div className="grid grid-cols-3 gap-2 mt-2">
              <button
                onClick={() => { onChangeHub(HubType.PLAYGROUND); setIsOpen(false); }}
                className={`py-2 text-[10px] uppercase tracking-wider font-extrabold text-center rounded-lg border ${
                  currentHub === HubType.PLAYGROUND
                    ? "bg-[#ffe599] text-black border-black border-2"
                    : "bg-white/5 text-gray-400 border-gray-300/30"
                }`}
              >
                Playground
              </button>
              <button
                onClick={() => { onChangeHub(HubType.STUDIO); setIsOpen(false); }}
                className={`py-2 text-[10px] uppercase tracking-wider font-extrabold text-center rounded-lg border ${
                  currentHub === HubType.STUDIO
                    ? "bg-[#1a56db] text-white border-[#1a56db]"
                    : "bg-white/5 text-gray-400 border-gray-300/30"
                }`}
              >
                Studio
              </button>
              <button
                onClick={() => { onChangeHub(HubType.LIBRARY); setIsOpen(false); }}
                className={`py-2 text-[10px] uppercase tracking-wider font-extrabold text-center rounded-lg border ${
                  currentHub === HubType.LIBRARY
                    ? "bg-[#e56845] text-white border-[#e56845]"
                    : "bg-white/5 text-gray-400 border-gray-300/30"
                }`}
              >
                Library
              </button>
            </div>
          </div>

          <div className="border-t border-black/10 dark:border-white/5 pt-3">
            <span className="text-[10px] font-mono font-extrabold uppercase text-gray-500 tracking-wider">Navigate Section</span>
            <div className="flex flex-col gap-3 mt-3 text-sm font-bold uppercase tracking-wider">
              {currentHub === HubType.PLAYGROUND && (
                <>
                  <button onClick={() => handleScrollTo("retro-hero")} className="text-left py-1.5 border-b border-black/5 hover:text-amber-500">
                    Home
                  </button>
                  <button onClick={() => handleScrollTo("ai-doubt-solver-section")} className="text-left py-1.5 border-b border-black/5 hover:text-amber-500">
                    Doubt Solver
                  </button>
                </>
              )}

              {currentHub === HubType.STUDIO && (
                <>
                  <button onClick={() => handleScrollTo("studio-hub")} className="text-left py-1.5 border-b border-slate-100 hover:text-[#1a56db]">
                    Syllabus
                  </button>
                  <button onClick={() => handleScrollTo("projects-section")} className="text-left py-1.5 border-b border-slate-100 hover:text-[#1a56db]">
                    Projects
                  </button>
                </>
              )}

              {currentHub === HubType.LIBRARY && (
                <>
                  <button onClick={() => handleScrollTo("library-hub")} className="text-[#3c3a3f] text-left py-1.5 border-b border-[#eae3d9] hover:text-[#e56845]">
                    Library
                  </button>
                  <button onClick={() => handleScrollTo("videos-section")} className="text-[#3c3a3f] text-left py-1.5 border-b border-[#eae3d9] hover:text-[#e56845]">
                    Videos
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
