import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { HubType } from "./types";
import Navbar from "./components/Navbar";
import PlaygroundHub from "./components/PlaygroundHub";
import StudioHub from "./components/StudioHub";
import LibraryHub from "./components/LibraryHub";
import { Github, Linkedin, Heart, Brain, Rocket } from "lucide-react";

export default function App() {
  const [currentHub, setCurrentHub] = useState<HubType>(HubType.PLAYGROUND);

  // Smooth scroll helper to top on page or hub changes to prevent jarring screen coordinates
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [currentHub]);

  // Handle footer colors matching each active theme precisely
  const getFooterStyles = () => {
    switch (currentHub) {
      case HubType.PLAYGROUND:
        return "bg-[#dfdad6] text-[#2c2b30] border-t-4 border-black font-sans";
      case HubType.STUDIO:
        return "bg-[#f8fafc] text-slate-500 border-t border-slate-200/80 font-sans";
      case HubType.LIBRARY:
        return "bg-[#f2efe8] text-[#2c2b30] border-t border-[#eae3d9] font-serif";
    }
  };

  return (
    <div className={`min-h-screen flex flex-col justify-between overflow-x-hidden transition-colors duration-500 ${
      currentHub === HubType.PLAYGROUND
        ? "bg-[#ece9e6]"
        : currentHub === HubType.STUDIO
        ? "bg-[#f1f5f9]"
        : "bg-[#faf7f2]"
    }`}>
      
      {/* GLOBAL NAVBAR PORTAL */}
      <Navbar currentHub={currentHub} onChangeHub={setCurrentHub} />

      {/* DYNAMIC CHAMBERS SWITCHER PANEL */}
      <main className="flex-grow">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentHub}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
          >
            {currentHub === HubType.PLAYGROUND && (
              <PlaygroundHub
                onNavigateToStudio={() => setCurrentHub(HubType.STUDIO)}
                onNavigateToLibrary={() => setCurrentHub(HubType.LIBRARY)}
              />
            )}
            
            {currentHub === HubType.STUDIO && <StudioHub />}

            {currentHub === HubType.LIBRARY && <LibraryHub />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* CUSTOM ADAPTING FOOTER PORTAL */}
      <footer className={`py-12 px-4 sm:px-6 lg:px-8 mt-auto ${getFooterStyles()}`}>
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          
          {/* Tagline & Logo */}
          <div className="flex flex-col items-center md:items-start text-center md:text-left gap-2 animate-fade-in">
            <span className="font-extrabold text-lg flex items-center gap-1.5 select-none text-current">
              <Brain className="w-5 h-5 text-indigo-500 animate-pulse" />
              <Rocket className="w-4 h-4 text-[#e56845] transform rotate-45" />
              <span>AI Launchpad</span>
            </span>
            <p className="text-xs max-w-sm font-semibold opacity-75">
              The premier knowledge base, roadmap studio, and code incubator created specifically 
              for modern AI/ML students.
            </p>
          </div>

          {/* Social Platform Placeholders */}
          <div className="flex items-center gap-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 ml-1 rounded-full bg-black/5 hover:bg-black/15 hover:scale-105 active:scale-95 transition-all text-current shadow-sm"
              title="GitHub Repository Placeholder"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="https://www.linkedin.com/in/darsh-nandu-072a39373"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-full bg-black/5 hover:bg-black/15 hover:scale-105 active:scale-95 transition-all text-current shadow-sm"
              title="Darsh Nandu's LinkedIn Profile"
            >
              <Linkedin className="w-5 h-5" />
            </a>
          </div>

          {/* Author Credits & Rights */}
          <div className="text-center md:text-right flex flex-col gap-1">
            <p className="text-xs font-bold uppercase tracking-wider flex items-center justify-center md:justify-end gap-1">
              <span>Built with</span>
              <Heart className="w-3.5 h-3.5 text-red-500 fill-red-500 animate-pulse inline-block" />
              <span>by</span>
              <a
                href="https://www.linkedin.com/in/darsh-nandu-072a39373"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-[#3d85c6] transition-colors cursor-pointer"
              >
                Darsh Nandu
              </a>
            </p>
            <p className="text-[10px] font-mono tracking-widest uppercase opacity-60">
              © Copyright 2025. All Rights Reserved.
            </p>
          </div>

        </div>
      </footer>

    </div>
  );
}
