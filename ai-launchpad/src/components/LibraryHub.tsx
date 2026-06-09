import React, { useState } from "react";
import { motion } from "motion/react";
import { BookOpen, ExternalLink, Library, Newspaper, Play, Settings } from "lucide-react";
import { COURSES, BOOKS, PAPERS, TOOLS, VIDEOS } from "../data";
import { ResourceItem } from "../types";

type TabOption = "Courses" | "Books" | "Papers" | "Tools";

export default function LibraryHub() {
  const [activeTab, setActiveTab] = useState<TabOption>("Courses");

  // Filter resources depending on current tab
  const getFilteredResources = (): ResourceItem[] => {
    switch (activeTab) {
      case "Courses":
        return COURSES;
      case "Books":
        return BOOKS;
      case "Papers":
        return PAPERS;
      case "Tools":
        return TOOLS;
    }
  };

  // Pre-determined editorial gradients and cover weights for the Stripe Press 3D Book Renders
  const getBookCoverClass = (index: number) => {
    const designs = [
      "from-rose-700 via-pink-600 to-rose-900 border-rose-400 text-rose-100",
      "from-amber-600 via-orange-500 to-amber-800 border-amber-400 text-amber-100",
      "from-teal-700 via-emerald-600 to-teal-900 border-teal-400 text-teal-100",
      "from-indigo-800 via-blue-700 to-indigo-950 border-indigo-400 text-indigo-100",
      "from-gray-800 via-slate-700 to-black border-slate-500 text-slate-100",
    ];
    return designs[index % designs.length];
  };

  return (
    <div id="library-hub" className="min-h-screen bg-[#faf7f2] text-[#2c2b30] pt-20 pb-24 relative overflow-hidden font-sans">
      
      {/* Stripe Press-like left navigation border and floating indicators */}
      <div className="absolute top-1/4 left-6 hidden xl:flex flex-col items-center gap-3 z-20">
        <div className="text-[10px] font-mono tracking-widest text-[#8a8580] rotate-270 translate-y-[-20px] select-none uppercase font-bold">
          Stripe Press Style
        </div>
        {[1, 2, 3, 4, 5, 6].map((dot) => (
          <div
            key={dot}
            className={`w-1.5 h-8 border border-[#c4c1bc] transition-colors rounded-sm ${
              dot === 1 ? "bg-[#e56845]" : "bg-transparent"
            }`}
          />
        ))}
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 mt-12 relative z-10">
        
        {/* EDITORIAL STRIPE HERO */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center mb-20">
          
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 bg-[#eae3d9]/60 px-3 py-1 rounded-full text-xs font-mono font-bold uppercase tracking-wider text-[#e56845] border border-[#d6cbbe]">
              <Library className="w-3.5 h-3.5" />
              <span>Darsh Nandu's Editorial Archives</span>
            </div>

            <h1 className="text-4xl md:text-6xl font-serif italic font-medium leading-tight tracking-tight text-[#2c2b30]">
              Hand-Picked <br className="hidden md:block"/>
              <span className="text-[#e56845] font-extrabold not-italic">Resources & Materials</span>
            </h1>

            <p className="text-[#646268] max-w-xl font-medium leading-relaxed text-base md:text-lg">
              Education is defined by the quality of index models you ingest. 
              Here resides a structured assembly of curriculum courses, theoretical manifestos, 
              original research papers, and technical platforms curated for AI students.
            </p>
          </div>

          <div className="lg:col-span-5 hidden lg:flex justify-center items-center">
            {/* Visual stacked book covers with 3D projection for Stripe Press feel */}
            <div className="relative w-72 h-80 perspective-[1000px]">
              
              {/* Back Book */}
              <div className="absolute top-0 left-6 w-52 h-72 bg-gradient-to-r from-blue-700 to-indigo-950 rounded-r-lg shadow-xl border-l-8 border-blue-500 transform rotate-y-[-15deg] translate-z-10 z-0 flex flex-col justify-between p-4 text-white">
                <span className="font-mono text-[9px] tracking-widest text-blue-200 uppercase">Book II</span>
                <span className="font-serif font-bold text-lg leading-tight">Deep Learning</span>
                <span className="text-[10px] font-mono text-blue-300">Goodfellow et al.</span>
              </div>

              {/* Middle Book */}
              <div className="absolute top-4 left-10 w-52 h-72 bg-gradient-to-r from-amber-600 to-orange-800 rounded-r-lg shadow-2xl border-l-8 border-amber-400 transform rotate-y-[-18deg] translate-z-20 z-10 flex flex-col justify-between p-4 text-white">
                <span className="font-mono text-[9px] tracking-widest text-amber-200 uppercase">Manual I</span>
                <span className="font-serif font-bold text-base leading-tight">Hands-On ML & AI</span>
                <span className="text-[10px] font-mono text-amber-200">Aurélien Géron</span>
              </div>

              {/* Front Book */}
              <div className="absolute top-8 left-16 w-52 h-72 bg-gradient-to-r from-rose-700 to-red-900 rounded-r-lg shadow-2xl border-l-8 border-rose-500 transform rotate-y-[-22deg] hover:rotate-y-[-5deg] transition-transform duration-500 translate-z-30 z-20 flex flex-col justify-between p-4 text-white hover:cursor-grab">
                <span className="font-mono text-[9px] tracking-widest text-rose-200 uppercase">Historic</span>
                <span className="font-serif font-bold text-lg leading-tight">Attention Is All You Need</span>
                <span className="text-[10px] font-mono text-rose-300">Google Brain 2017</span>
              </div>

            </div>
          </div>

        </div>

        {/* TAB SWITCHER AND CARD INDEX GRID */}
        <div className="bg-white/40 border border-[#eae3d9] p-6 rounded-3xl shadow-xl mb-20 backdrop-blur-sm">
          
          {/* TAB HEADER */}
          <div className="border-b border-[#eae3d9] pb-4 mb-8 flex flex-wrap gap-2 justify-between items-center">
            
            <h3 className="font-serif italic text-2xl font-semibold text-[#2c2b30]">
              The Scholar's Desk
            </h3>

            <div className="flex overflow-x-auto max-w-full gap-1 p-0.5 sm:p-1 bg-[#eae3d9]/40 border border-[#d6cbbe] rounded-xl scrollbar-none">
              {(["Courses", "Books", "Papers", "Tools"] as TabOption[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-2.5 sm:px-4 py-1.5 sm:py-2 rounded-md sm:rounded-lg text-xs sm:text-sm font-bold transition-all uppercase tracking-wide cursor-pointer whitespace-nowrap ${
                    activeTab === tab
                      ? "bg-[#e56845] text-white shadow"
                      : "text-gray-650 text-gray-600 hover:bg-[#eae3d9]/70 hover:text-black"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

          </div>

          {/* FEATURED: Research Paper Finder — shown only on Papers tab */}
          {activeTab === "Papers" && (
            <a
              href="https://darsh-nandu.github.io/scientific-ledger/"
              target="_blank"
              rel="noopener noreferrer"
              className="group block w-full mb-8 rounded-2xl border-2 border-[#2c2b30] overflow-hidden shadow-[4px_4px_0px_0px_rgba(44,43,48,1)] hover:shadow-[6px_6px_0px_0px_rgba(229,104,69,1)] transition-all duration-300 hover:-translate-y-0.5"
            >
              <div className="bg-[#2c2b30] px-6 py-4 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="bg-[#e56845] w-9 h-9 rounded-xl border-2 border-[#e56845]/40 flex items-center justify-center shrink-0">
                    <Newspaper className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <p className="text-[10px] font-mono font-bold uppercase tracking-widest text-[#e56845] mb-0.5">
                      ✦ Featured Tool
                    </p>
                    <h3 className="font-serif font-bold text-white text-lg leading-tight">
                      Scientific Ledger — Research Paper Finder
                    </h3>
                  </div>
                </div>
                <span className="shrink-0 text-[11px] font-mono font-bold uppercase tracking-wider text-[#e56845] flex items-center gap-1 group-hover:translate-x-1 transition-transform duration-150">
                  Open <ExternalLink className="w-3 h-3" />
                </span>
              </div>
              <div className="bg-[#f7f3ee] px-6 py-3 flex items-center justify-between gap-4">
                <p className="text-sm text-gray-600 font-medium leading-relaxed">
                  My personal research discovery tool — search, filter, and explore academic papers across AI, ML, and Computer Science. Built for students who are tired of drowning in Google Scholar noise.
                </p>
                <div className="hidden sm:flex gap-2 shrink-0">
                  {["Search", "Filter", "Discover"].map((tag) => (
                    <span key={tag} className="text-[10px] font-mono bg-[#e56845]/10 text-[#e56845] border border-[#e56845]/20 px-2 py-0.5 rounded font-bold uppercase tracking-wider">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </a>
          )}

          {/* DYNAMIC CARD RENDERS */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {getFilteredResources().map((res, index) => {
              const isBookTab = activeTab === "Books" || activeTab === "Papers";
              
              return (
                <div
                  key={res.id}
                  className="bg-white border border-[#eae3d9] hover:border-[#d6cbbe] rounded-2xl p-6 transition-all duration-300 shadow-md hover:shadow-xl hover:-translate-y-1 flex flex-col justify-between"
                >
                  {isBookTab ? (
                    // 3D Visual Book preview row if we represent books or papers
                    <div className="flex gap-4 items-start mb-4">
                      {/* CSS 3D Book spine block simulation */}
                      <div className={`w-9 h-14 bg-gradient-to-r flex-shrink-0 rounded-sm border-l-4 shadow flex flex-col justify-end p-1 font-bold select-none ${getBookCoverClass(index)}`}>
                        <span className="text-[5px] uppercase tracking-widest font-mono text-center mb-1">AI</span>
                      </div>
                      <div>
                        <h4 className="font-serif text-lg font-bold text-[#2c2b30] leading-snug line-clamp-2">
                          {res.name}
                        </h4>
                      </div>
                    </div>
                  ) : (
                    <div className="mb-4">
                      <h4 className="font-serif text-lg font-bold text-[#2c2b30] leading-snug">
                        {res.name}
                      </h4>
                    </div>
                  )}

                  <div className="space-y-4">
                    {!isBookTab && (
                      <p className="text-sm text-gray-600 leading-relaxed min-h-[48px] line-clamp-3">
                        {res.description}
                      </p>
                    )}
                    {isBookTab && (
                      <p className="text-xs text-gray-600 leading-relaxed font-semibold italic min-h-[40px] line-clamp-3">
                        {res.description}
                      </p>
                    )}

                    <div className="flex flex-wrap gap-1.5 pt-2">
                      {res.tags.map((tag, tIdx) => (
                        <span
                          key={tIdx}
                          className="text-[10px] font-mono bg-amber-100/50 text-[#855e3a] border border-amber-200/40 px-2 py-0.5 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>

                    <div className="pt-3 border-t border-gray-100 flex items-center justify-between">
                      <a
                        href={res.url}
                        target="_blank"
                        rel="referrer"
                        className="text-xs font-mono font-bold tracking-wider text-[#e56845] hover:text-[#c45330] inline-flex items-center gap-1 uppercase"
                      >
                        Visit Resource <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

        </div>

        {/* YOUTUBE MEDIA & PLAYLISTS SECTION */}
        <div id="videos-section" className="border-t border-[#eae3d9] pt-16">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-[#e56845]/10 border border-[#e56845]/20 rounded-full text-xs font-mono text-[#e56845] font-bold uppercase tracking-wider mb-4">
              <Play className="w-3.5 h-3.5 fill-[#e56845]" />
              <span>Masterclass Video Portal</span>
            </div>
            
            <h2 className="text-3xl md:text-5xl font-serif font-medium text-[#2c2b30] tracking-tight mb-4">
              Recommended Videos & Playlists
            </h2>
            <p className="text-gray-600 max-w-xl mx-auto font-medium leading-relaxed">
              Handpicked by me so you don't waste hours searching through basic online bootcamps.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {VIDEOS.map((video) => (
              <a
                key={video.id}
                href={
                  video.playlistId
                    ? `https://www.youtube.com/playlist?list=${video.playlistId}`
                    : `https://www.youtube.com/watch?v=${video.youtubeId}`
                }
                target="_blank"
                rel="noopener noreferrer"
                className="group bg-white border border-[#eae3d9] rounded-2xl overflow-hidden hover:shadow-2xl transition-all flex flex-col justify-between"
              >
                <div>
                  
                  {/* YouTube Thumbnail Frame inside high-end borders */}
                  <div className="relative aspect-video bg-[#2c2b30] overflow-hidden border-b border-[#eae3d9]">
                    <img
                      src={`https://img.youtube.com/vi/${video.youtubeId}/hqdefault.jpg`}
                      alt={video.title}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500 opacity-90 group-hover:opacity-100"
                    />
                    <div className="absolute inset-0 bg-black/20 group-hover:bg-transparent transition-colors flex items-center justify-center">
                      <div className="bg-[#e56845] w-12 h-12 rounded-full border-2 border-white flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform">
                        <Play className="w-5 h-5 text-white fill-white ml-0.5" />
                      </div>
                    </div>

                    <span className="absolute bottom-3 right-3 bg-black/80 text-white font-mono text-[10px] px-2 py-0.5 uppercase tracking-wider font-bold rounded">
                      {video.badge}
                    </span>
                  </div>

                  <div className="p-5">
                    <span className="font-mono text-[10px] text-gray-500 uppercase tracking-widest font-bold">
                      {video.channel}
                    </span>
                    <h3 className="font-serif font-bold text-base text-[#2c2b30] mt-1 leading-snug line-clamp-2 duration-200 group-hover:text-[#e56845]">
                      {video.title}
                    </h3>
                  </div>

                </div>

                <div className="p-5 pt-0 flex justify-end">
                  <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#e56845] flex items-center gap-1 group-hover:translate-x-1 duration-150">
                    Open on YouTube <ExternalLink className="w-3 h-3" />
                  </span>
                </div>
              </a>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
