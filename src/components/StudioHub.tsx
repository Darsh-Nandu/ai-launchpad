import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Brain, Cpu, MessageSquare, Eye, Sparkles, Clock, Compass, ArrowRight,
  BookOpen, Code, Shield, Layers, ChevronRight, ChevronLeft, RotateCcw, 
  Check, Play, Sliders, Activity, Sparkle, Target, CheckCircle2
} from "lucide-react";
import { ROADMAP_TRACKS, PROJECTS } from "../data";
import { RoadmapTrack, ProjectItem, Difficulty } from "../types";

// Core ambient light gradients that change periodically to keep the experience active and fresh
const AMBIENT_GRADIENTS = [
  // 1. Fresh Minty Emerald (ML Fundamentals Theme)
  {
    bg: "bg-gradient-to-tr from-[#f0fdf4] via-[#f7fee7] to-[#FAF9F6]",
    glow1: "bg-[#86efac]/25",
    glow2: "bg-[#99f6e4]/30",
    glow3: "bg-[#ccfbf1]/25",
    glow4: "bg-[#fef08a]/20"
  },
  // 2. High-end lavender & violet (Deep Learning Theme)
  {
    bg: "bg-gradient-to-tr from-[#f5f3ff] via-[#eef2ff] to-[#FAF9F6]",
    glow1: "bg-[#c7d2fe]/30",
    glow2: "bg-[#ddd6fe]/30",
    glow3: "bg-[#f3e8ff]/25",
    glow4: "bg-[#e0e7ff]/20"
  },
  // 3. Crisp ocean air cyan (NLP Theme)
  {
    bg: "bg-gradient-to-tr from-[#f0f9ff] via-[#e0f2fe] to-[#FAF9F6]",
    glow1: "bg-[#bae6fd]/35",
    glow2: "bg-[#fed7aa]/25",
    glow3: "bg-[#e0f2fe]/30",
    glow4: "bg-[#dbeafe]/20"
  },
  // 4. Cyber blossom pink orchid (Computer Vision Theme)
  {
    bg: "bg-gradient-to-tr from-[#faf5ff] via-[#fdf2f8] to-[#FAF9F6]",
    glow1: "bg-[#e9d5ff]/30",
    glow2: "bg-[#fbcfe8]/35",
    glow3: "bg-[#fae8ff]/25",
    glow4: "bg-[#fbcfe8]/20"
  },
  // 5. Sunny warm sunset apricot (Generative AI Theme)
  {
    bg: "bg-gradient-to-tr from-[#fff7ed] via-[#fef3c7] to-[#FAF9F6]",
    glow1: "bg-[#fed7aa]/30",
    glow2: "bg-[#fef08a]/35",
    glow3: "bg-[#ffe4e6]/25",
    glow4: "bg-[#fef9c3]/20"
  },
  // 6. Calming soft rose cream
  {
    bg: "bg-gradient-to-tr from-[#fff1f2] via-[#fef2f2] to-[#FAF9F6]",
    glow1: "bg-[#fecdd3]/20",
    glow2: "bg-[#fce7f3]/25",
    glow3: "bg-[#ffe4e6]/20",
    glow4: "bg-[#f1f5f9]/15"
  },
  // 7. Dewy grassland emerald
  {
    bg: "bg-gradient-to-tr from-[#ecfdf5] via-[#f0fdf4] to-[#FAF9F6]",
    glow1: "bg-[#a7f3d0]/25",
    glow2: "bg-[#cbd5e1]/20",
    glow3: "bg-[#cbd5e1]/20",
    glow4: "bg-[#fefcb3]/15"
  }
];

export default function StudioHub() {
  const [selectedTrack, setSelectedTrack] = useState<RoadmapTrack>(ROADMAP_TRACKS[0]);
  const [completedTopics, setCompletedTopics] = useState<Record<string, boolean>>(() => {
    try {
      const saved = localStorage.getItem("ai_launchpad_completed_topics");
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  // Project Sandbox & Panel State
  const [activeDifficulty, setActiveDifficulty] = useState<Difficulty>("Beginner");
  const [selectedProject, setSelectedProject] = useState<ProjectItem | null>(PROJECTS[0]);
  const [selectedColorIndex, setSelectedColorIndex] = useState<Record<string, number>>({
    "ml-fundamentals": 0,
    "deep-learning": 0,
    "nlp": 0,
    "computer-vision": 0,
    "generative-ai": 0
  });

  // Sandbox inputs
  const [sandboxInputs, setSandboxInputs] = useState<Record<string, any>>({});
  const [sandboxOutputs, setSandboxOutputs] = useState<any>(null);
  const [sandboxRunning, setSandboxRunning] = useState<boolean>(false);
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);

  const carouselRef = useRef<HTMLDivElement>(null);
  const projectsScrollRef = useRef<HTMLDivElement>(null);

  const [bgIndex, setBgIndex] = useState(0);

  // Periodically change background color gradients every 7 seconds
  useEffect(() => {
    const timer = setInterval(() => {
      setBgIndex((prev) => (prev + 1) % AMBIENT_GRADIENTS.length);
    }, 7000);
    return () => clearInterval(timer);
  }, []);

  const handleTrackSelect = (track: RoadmapTrack) => {
    setSelectedTrack(track);
    const trackIndexMap: Record<string, number> = {
      "ml-fundamentals": 0,
      "deep-learning": 1,
      "nlp": 2,
      "computer-vision": 3,
      "generative-ai": 4
    };
    if (track.id in trackIndexMap) {
      setBgIndex(trackIndexMap[track.id]);
    }
  };

  // Sync completion states with localStorage
  useEffect(() => {
    localStorage.setItem("ai_launchpad_completed_topics", JSON.stringify(completedTopics));
  }, [completedTopics]);

  // Sync default selected project when difficulty tabs change
  useEffect(() => {
    const filtered = PROJECTS.filter((p) => p.difficulty === activeDifficulty);
    if (filtered.length > 0) {
      setSelectedProject(filtered[0]);
      setSandboxOutputs(null);
      setGeneratedTokens([]);
    }
  }, [activeDifficulty]);

  // Handle default inputs when current project is loaded
  useEffect(() => {
    if (selectedProject) {
      setSandboxOutputs(null);
      setGeneratedTokens([]);
      // Default sandbox states to match sliders
      switch (selectedProject.name) {
        case "Diabetes Risk Estimator":
          setSandboxInputs({ glucose: 120, bmi: 24, age: 38 });
          break;
        case "House Prices Predictor (Kaggle Classic)":
          setSandboxInputs({ sqft: 1800, beds: 3, tier: "Suburban" });
          break;
        case "Customer Clustered Segments with K-Means":
          setSandboxInputs({ recency: 12, frequency: 6, clusters: 4 });
          break;
        case "Interactive Spambox Classifier":
          setSandboxInputs({ text: "URGENT: Win an award today now! Cash bonuses await!!" });
          break;
        default:
          setSandboxInputs({ lr: 0.01, epochs: 50, batchSize: 32 });
          break;
      }
    }
  }, [selectedProject]);

  const toggleTopicCompletion = (key: string) => {
    setCompletedTopics((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const getTrackStatistics = (track: RoadmapTrack) => {
    let total = 0, completed = 0;
    track.syllabus.forEach((week, wIdx) => {
      week.topics.forEach((_, tIdx) => {
        total++;
        if (completedTopics[`${track.id}-${wIdx}-${tIdx}`]) completed++;
      });
    });
    return { total, completed, percentage: total > 0 ? Math.round((completed / total) * 100) : 0 };
  };

  const resetTrackProgress = (trackId: string) => {
    setCompletedTopics((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((key) => {
        if (key.startsWith(`${trackId}-`)) delete copy[key];
      });
      return copy;
    });
  };

  // Color preset options optimized for beautiful premium Light layout
  const getTrackColorPresets = (id: string, colorIdx: number = 0) => {
    const presets: Record<string, Array<{ themeColor: string; name: string; bannerBg: string; colorClass: string; dots: string[]; shortText: string }>> = {
      "ml-fundamentals": [
        { themeColor: "#10b981", name: "Emerald Mint", bannerBg: "from-emerald-100/40 to-teal-50/50", colorClass: "text-emerald-700 bg-emerald-50 border-emerald-250/20", dots: ["#10b981", "#14b8a6"], shortText: "Explore parameters, gradient descents, & regression models." },
        { themeColor: "#0ea5e9", name: "Oxygen Sky", bannerBg: "from-sky-100/40 to-slate-50/40", colorClass: "text-sky-700 bg-sky-50 border-sky-200/25", dots: ["#0ea5e9", "#06b6d4"], shortText: "Explore dynamic clustering & continuous projection models." }
      ],
      "deep-learning": [
        { themeColor: "#6366f1", name: "Lavender Glass", bannerBg: "from-indigo-100/40 to-violet-50/50", colorClass: "text-indigo-700 bg-indigo-50 border-indigo-200/25", dots: ["#6366f1", "#a855f7"], shortText: "Construct neural chains with weight optimization blocks." },
        { themeColor: "#a855f7", name: "Prestige Violet", bannerBg: "from-fuchsia-100/40 to-indigo-50/40", colorClass: "text-purple-700 bg-purple-50 border-purple-200/25", dots: ["#a855f7", "#ec4899"], shortText: "Execute straightforward backpropagation steps in PyTorch." }
      ],
      "nlp": [
        { themeColor: "#0ea5e9", name: "Oxygen Blue", bannerBg: "from-sky-100/40 to-slate-100/40", colorClass: "text-sky-700 bg-sky-50 border-sky-200/25", dots: ["#0ea5e9", "#3b82f6"], shortText: "Understand bag-of-words text arrays and modern attention mechanisms." },
        { themeColor: "#3b82f6", name: "Deep Cobalt", bannerBg: "from-blue-100/40 to-slate-50/40", colorClass: "text-blue-700 bg-blue-50 border-blue-200/25", dots: ["#3b82f6", "#0ea5e9"], shortText: "Map vocabularies using pre-trained word tokens easily." }
      ],
      "computer-vision": [
        { themeColor: "#a855f7", name: "Grape Frost", bannerBg: "from-purple-100/40 to-pink-50/40", colorClass: "text-purple-700 bg-purple-50 border-purple-200/25", dots: ["#a855f7", "#ec4899"], shortText: "Isolate target visual matrices with precise outline detections." },
        { themeColor: "#ec4899", name: "Cyber Blossom", bannerBg: "from-pink-100/40 to-purple-50/40", colorClass: "text-pink-700 bg-pink-50 border-pink-200/25", dots: ["#ec4899", "#0ea5e9"], shortText: "Learn YOLO object trackers and bounding frameworks cleanly." }
      ],
      "generative-ai": [
        { themeColor: "#f97316", name: "Honey Fusion", bannerBg: "from-orange-100/40 to-amber-50/50", colorClass: "text-orange-700 bg-orange-50 border-orange-200/25", dots: ["#f97316", "#ef4444"], shortText: "Integrate powerful models with clean developer APIs & prompt layers." },
        { themeColor: "#ef4444", name: "Amber Flare", bannerBg: "from-red-100/40 to-amber-50/50", colorClass: "text-red-700 bg-red-50 border-red-200/25", dots: ["#ef4444", "#eed202"], shortText: "Assemble multi-agent workflows and real-time response chains." }
      ]
    };
    const presetList = presets[id] || [{ themeColor: "#64748b", name: "Classic Metal", bannerBg: "from-slate-100/50 to-slate-200/50", colorClass: "text-slate-700 border-slate-200/25", dots: ["#64748b"], shortText: "Structured training steps for modern machine learning foundations." }];
    return presetList[colorIdx % presetList.length];
  };

  const executeSandbox = () => {
    setSandboxRunning(true);
    setSandboxOutputs(null);
    setGeneratedTokens([]);

    if (selectedProject?.name === "Mini-GPT Transformer model from scratch" || selectedProject?.name === "LSTM Character-Level Language Generator") {
      let tokens = ["The", " self", "-attention", " mechanism", " maps", " input", " context", " vectors", " to", " predict", " next", " logical", " text", " sequences", " with", " high", " accuracy", " and", " low", " prediction", " error", "."];
      let current: string[] = [];
      let i = 0;
      const interval = setInterval(() => {
        if (i < tokens.length) {
          current.push(tokens[i]);
          setGeneratedTokens([...current]);
          i++;
        } else {
          clearInterval(interval);
          setSandboxRunning(false);
          setSandboxOutputs({ status: "Success", perplexity: "4.12", duration: "110ms" });
        }
      }, 60);
    } else {
      setTimeout(() => {
        setSandboxRunning(false);
        if (selectedProject?.name === "Diabetes Risk Estimator") {
          const glucose = sandboxInputs.glucose || 120;
          const bmi = sandboxInputs.bmi || 24;
          const age = sandboxInputs.age || 38;
          const riskVal = Math.round((glucose * 0.35 + bmi * 0.8 + age * 0.15));
          setSandboxOutputs({
            metricName: "Estimated Risk Probability",
            metricValue: `${Math.min(99, Math.max(1, riskVal))}%`,
            classification: riskVal > 80 ? "Elevated Warning (Consult Clinician)" : riskVal > 50 ? "Moderate Alert (Requires Monitoring)" : "Optimal Healthy Level",
            note: "Calculated via continuous logistic formula extrapolation inputs"
          });
        } else if (selectedProject?.name === "House Prices Predictor (Kaggle Classic)") {
          const sqft = sandboxInputs.sqft || 1800;
          const price = Math.round((sqft * 145 + 3 * 22000) * 1.12);
          setSandboxOutputs({
            metricName: "Predicted Valuation Estimate",
            metricValue: `₹ ${(price * 1.5).toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
            classification: "Optimal Market Projection Match",
            note: "Determined using continuous surface-area linear slope gradients"
          });
        } else if (selectedProject?.name === "Customer Clustered Segments with K-Means") {
          const clusters = sandboxInputs.clusters || 4;
          setSandboxOutputs({
            metricName: "Formed Consumer Clusters",
            metricValue: `${clusters} Distinct Groups`,
            classification: "Balanced Cohort Segment Partition",
            note: "Calculated with Euclidean distance calculations separating inputs"
          });
        } else if (selectedProject?.name === "Interactive Spambox Classifier") {
          const text = (sandboxInputs.text || "").toLowerCase();
          const hasSpam = /urgent|win|money|cash|award|gift|claim|free/.test(text);
          const conf = hasSpam ? 89.2 : 12.1;
          setSandboxOutputs({
            metricName: "Spam Flag Probability",
            metricValue: `${conf}%`,
            classification: hasSpam ? "SPAM ALERT DETECTED" : "REGULAR PERSONAL EMAIL",
            note: "Triggered via basic vocabulary indicator word frequencies"
          });
        } else {
          const epochs = sandboxInputs.epochs || 50;
          const finalLoss = (0.95 / (0.01 * epochs * 4 + 1)).toFixed(5);
          setSandboxOutputs({
            metricName: "Mathematical Model Error",
            metricValue: finalLoss,
            classification: "Converged Graph Solver state",
            note: "Loss values minimized continuously through mock iterations"
          });
        }
      }, 800);
    }
  };

  const scrollTracks = (direction: "left" | "right") => {
    if (carouselRef.current) {
      const { scrollLeft, clientWidth } = carouselRef.current;
      carouselRef.current.scrollTo({
        left: scrollLeft + (direction === "left" ? -clientWidth * 0.7 : clientWidth * 0.7),
        behavior: "smooth"
      });
    }
  };

  // Get background styles dynamically matching the active ambient gradient index
  const backdrop = AMBIENT_GRADIENTS[bgIndex];

  return (
    <div id="studio-hub" className={`min-h-screen text-slate-800 pt-24 pb-36 font-sans antialiased relative overflow-hidden transition-all duration-[1200ms] ${backdrop.bg}`}>
      
      {/* ─── Apple Liquid Holographic Backglowing Pools ─── */}
      <div className={`absolute top-[-5%] left-[-10%] w-[55rem] h-[55rem] rounded-full blur-[140px] pointer-events-none opacity-60 transition-all duration-[1200ms] ${backdrop.glow1}`} />
      <div className={`absolute top-[25%] right-[-15%] w-[45rem] h-[45rem] rounded-full blur-[160px] pointer-events-none opacity-50 transition-all duration-[1200ms] ${backdrop.glow2}`} />
      <div className={`absolute bottom-[5%] left-[5%] w-[50rem] h-[50rem] rounded-full blur-[150px] pointer-events-none opacity-50 transition-all duration-[1200ms] ${backdrop.glow3}`} />
      <div className={`absolute top-[50%] left-[25%] w-[40rem] h-[40rem] rounded-full blur-[130px] pointer-events-none opacity-40 transition-all duration-[1200ms] ${backdrop.glow4}`} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">

        {/* ─── Apple Header: Interactive Showcase line-up ─── */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 mt-6">
          <div>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-white border border-slate-200 rounded-full text-[10px] font-mono font-bold text-slate-600 mb-3 uppercase tracking-wide shadow-sm">
              <Sparkles className="w-3.5 h-3.5 text-sky-500" /> Liquid Glass Experience
            </span>
            <h1 className="text-4xl md:text-5.5xl font-black text-slate-900 tracking-tight leading-none">
              Explore the line-up.
            </h1>
            <p className="text-slate-500 text-sm md:text-base font-semibold mt-3 max-w-xl">
              Select an advanced AI/ML track to customize its background matrices. Click the color finishes below to dynamically shift ambient pigments.
            </p>
          </div>

          <div className="mt-4 md:mt-0">
            <button
              onClick={() => document.getElementById("active-syllabus-stage")?.scrollIntoView({ behavior: "smooth" })}
              className="group text-sky-600 hover:text-sky-700 font-bold text-sm flex items-center gap-1.5 cursor-pointer bg-white py-2.5 px-4.5 rounded-xl border border-slate-200/80 shadow-sm hover:shadow-md transition-all"
            >
              <span>Compare syllabus</span>
              <ChevronRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
            </button>
          </div>
        </div>

        {/* ─── Apple Color Customized Interactive Track Carousel ─── */}
        <div className="relative group/carousel">
          <button
            onClick={() => scrollTracks("left")}
            className="absolute -left-4 top-[40%] -translate-y-1/2 z-20 w-11 h-11 rounded-full backdrop-blur-xl bg-white/90 border border-slate-200 shadow-md hover:bg-slate-50 flex items-center justify-center text-slate-600 hover:text-slate-900 transition-all cursor-pointer"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <button
            onClick={() => scrollTracks("right")}
            className="absolute -right-4 top-[40%] -translate-y-1/2 z-20 w-11 h-11 rounded-full backdrop-blur-xl bg-white/90 border border-slate-200 shadow-md hover:bg-slate-50 flex items-center justify-center text-slate-600 hover:text-slate-900 transition-all cursor-pointer"
          >
            <ChevronRight className="w-5 h-5" />
          </button>

          <div
            ref={carouselRef}
            className="flex gap-6 overflow-x-auto snap-x snap-mandatory scrollbar-none pt-8 pb-14 px-2 scroll-smooth"
          >
            {ROADMAP_TRACKS.map((track) => {
              const isSelected = selectedTrack.id === track.id;
              const colorIdx = selectedColorIndex[track.id] || 0;
              const preset = getTrackColorPresets(track.id, colorIdx);
              const { percentage } = getTrackStatistics(track);

              return (
                <div
                  key={track.id}
                  onClick={() => handleTrackSelect(track)}
                  className={`snap-start shrink-0 w-[290px] sm:w-[350px] bg-white/60 backdrop-blur-xl border rounded-[30px] p-6 cursor-pointer select-none transition-all duration-300 relative flex flex-col justify-between ${
                    isSelected
                      ? "border-sky-500/80 shadow-[0_22px_45px_rgba(14,165,233,0.08),_inset_0_1.5px_1.5px_rgba(255,255,255,1)] bg-white/90 -translate-y-1.5 focus:scale-[1.01]"
                      : "border-slate-200/80 hover:border-slate-300 hover:bg-white/80 shadow-[0_8px_30px_rgba(0,0,0,0.01)]"
                  }`}
                >
                  {/* Glossy sweep */}
                  <div className="absolute inset-0 rounded-[30px] overflow-hidden pointer-events-none select-none">
                    <div className="absolute top-0 left-[-100%] w-[300%] h-[100%] bg-gradient-to-r from-transparent via-white/40 to-transparent skew-x-[-25deg] group-hover:left-[100%] transition-all duration-1000 ease-out" />
                  </div>

                  {/* Liquid Glass Mockup Box */}
                  <div className={`w-full h-44 rounded-2xl bg-gradient-to-b ${preset.bannerBg} border border-slate-200/50 p-4 flex flex-col justify-between relative overflow-hidden shadow-inner`}>
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 rounded-full bg-white/40 filter blur-2xl pointer-events-none select-none" />
                    
                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 font-bold relative z-10">
                      <span className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                        {preset.name}
                      </span>
                      <span>Level: {track.difficulty}</span>
                    </div>

                    {/* Simple Static Vector representation (No blinking dots) */}
                    <div className="h-20 w-full flex items-center justify-center relative z-10">
                      {track.id === "ml-fundamentals" && (
                        <svg className="w-4/5 h-14 overflow-visible">
                          <polyline fill="none" stroke={preset.themeColor} strokeWidth="3" strokeLinecap="round" strokeDasharray="4 3" points="10,40 45,30 80,15 120,45 160,20 200,35" />
                          <circle cx="160" cy="20" r="5" fill="#ffffff" className="stroke-2 shadow-md" style={{ stroke: preset.themeColor }} />
                          <circle cx="80" cy="15" r="4" fill={preset.themeColor} />
                        </svg>
                      )}
                      {track.id === "deep-learning" && (
                        <div className="flex gap-4 items-center">
                          <div className="flex flex-col gap-2.5">
                            <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
                            <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
                            <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
                          </div>
                          <div className="flex flex-col gap-2 relative">
                            <span className="w-3.5 h-3.5 rounded-full shadow-md" style={{ backgroundColor: preset.themeColor }} />
                            <span className="w-3.5 h-3.5 rounded-full bg-slate-300/60" />
                          </div>
                          <div className="flex flex-col gap-2.5">
                            <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
                          </div>
                        </div>
                      )}
                      {track.id === "nlp" && (
                        <div className="flex flex-col gap-1.5 items-center">
                          <span className="text-[10px] font-mono text-emerald-700 border border-emerald-300/30 px-2 py-0.5 rounded bg-emerald-50">Text Vector</span>
                          <span className="text-xs text-slate-400">➔</span>
                          <span className="text-[10px] font-mono text-cyan-700 border border-cyan-300/30 px-2 py-0.5 rounded bg-sky-50">Words Projection</span>
                        </div>
                      )}
                      {track.id === "computer-vision" && (
                        <div className="w-16 h-16 border-2 border-dashed rounded-lg flex items-center justify-center" style={{ borderColor: preset.themeColor }}>
                          <span className="w-8 h-1 bg-slate-800 opacity-40 animate-none" />
                        </div>
                      )}
                      {track.id === "generative-ai" && (
                        <div className="relative">
                          <Brain className="w-12 h-12 opacity-80" style={{ color: preset.themeColor }} />
                          <Sparkle className="w-5 h-5 absolute -right-2 -top-2 text-amber-500 animate-none" />
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between text-[9px] font-mono text-slate-400 relative z-10 pt-2 border-t border-slate-200/50">
                      <span>CHASSIS: {preset.name.toUpperCase()}</span>
                      <span>{percentage}% Done</span>
                    </div>
                  </div>

                  {/* Aesthetic Color Swappers */}
                  <div className="flex justify-center items-center gap-2 mt-4 mb-2">
                    {Array.from({ length: 2 }).map((_, pIdx) => {
                      const opt = getTrackColorPresets(track.id, pIdx);
                      return (
                        <button
                          key={pIdx}
                          title={opt.name}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedColorIndex(prev => ({ ...prev, [track.id]: pIdx }));
                          }}
                          className={`w-3.5 h-3.5 rounded-full border transition-all focus:outline-none ${
                            colorIdx === pIdx
                              ? "border-slate-500 scale-110 ring-2 ring-sky-300/60"
                              : "border-transparent opacity-60 hover:opacity-100"
                          }`}
                          style={{ backgroundColor: opt.themeColor }}
                        />
                      );
                    })}
                  </div>

                  {/* Details */}
                  <div className="mt-1 text-center">
                    <h3 className="font-extrabold text-base text-slate-900 tracking-tight leading-snug">
                      {track.title}
                    </h3>
                    <p className="text-slate-500 text-xs mt-2 leading-relaxed min-h-[40px] px-1 line-clamp-2">
                      {preset.shortText}
                    </p>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs font-mono text-slate-400">
                    <span>Intensity</span>
                    <span className="text-slate-700 font-extrabold">{track.duration}</span>
                  </div>

                  <button className="w-full mt-4 py-2 bg-gradient-to-b from-white to-slate-50 border border-slate-200/80 hover:border-slate-300 rounded-xl font-bold text-xs text-slate-700 transition-all cursor-pointer flex items-center justify-center gap-1.5 shadow-sm">
                    <span>Expand Syllabus</span> <ArrowRight className="w-3.5 h-3.5" />
                  </button>

                </div>
              );
            })}
          </div>
        </div>

        {/* ─── Active Syllabus Panel Detail (Liquid Glass Card) ─── */}
        <div id="active-syllabus-stage" className="mt-4 mb-24">
          <AnimatePresence mode="wait">
            <motion.div
              key={selectedTrack.id + (selectedColorIndex[selectedTrack.id] || 0)}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="backdrop-blur-2xl bg-white/70 border border-white/90 shadow-2xl rounded-[32px] p-6 sm:p-10 relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/80 to-transparent pointer-events-none" />
              
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-8 border-b border-slate-200/60 mb-8">
                <div>
                  <div className="flex items-center gap-1.5 text-sky-600 font-mono text-[10px] font-bold uppercase tracking-widest mb-2">
                    <Compass className="w-3.5 h-3.5" /> Learning Syllabus Console
                  </div>
                  <h2 className="text-2xl sm:text-3.5xl font-black text-slate-900 tracking-tight leading-tight">
                    {selectedTrack.title}
                  </h2>
                  <p className="text-slate-500 text-sm font-medium max-w-xl mt-1.5 leading-relaxed">
                    {selectedTrack.description}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  <div className="px-4 py-2 bg-slate-100/60 border border-slate-200/80 rounded-xl text-center">
                    <span className="text-[9px] uppercase tracking-wider font-mono text-slate-400 block font-bold leading-none mb-1">Duration</span>
                    <strong className="text-xs sm:text-sm font-mono text-slate-800 font-bold">{selectedTrack.duration}</strong>
                  </div>
                  <div className="px-4 py-2 bg-slate-100/60 border border-slate-200/80 rounded-xl text-center">
                    <span className="text-[9px] uppercase tracking-wider font-mono text-slate-400 block font-bold leading-none mb-1">Skill Rating</span>
                    <strong className="text-xs sm:text-sm font-mono text-sky-600 font-extrabold">{selectedTrack.difficulty}</strong>
                  </div>
                </div>
              </div>

              {/* Progress Tracker bar */}
              {(() => {
                const stats = getTrackStatistics(selectedTrack);
                return (
                  <div className="mb-4 p-4 bg-slate-50/50 border border-slate-200/60 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-4 shadow-inner">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-tr from-sky-400 to-indigo-500 flex items-center justify-center font-mono text-sm font-black text-white shadow-md">
                        {stats.percentage}%
                      </div>
                      <div>
                        <h4 className="text-xs sm:text-sm font-bold text-slate-800">Learning Progress tracker</h4>
                        <p className="text-[10px] text-slate-400 font-semibold font-mono">Progress updates automatically as you complete targets</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-start">
                      <span className="text-xs text-slate-500 font-mono font-bold">{stats.completed} of {stats.total} Topics complete</span>
                      {stats.percentage > 0 && (
                        <button
                          onClick={() => resetTrackProgress(selectedTrack.id)}
                          className="text-[10px] font-mono font-bold text-red-650 text-red-650 hover:text-red-600 bg-red-50/60 border border-red-100 px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.2 shadow-sm"
                        >
                          <RotateCcw className="w-3.5 h-3.5" /> Clear Progress
                        </button>
                      )}
                    </div>
                  </div>
                );
              })()}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                {selectedTrack.syllabus.map((week, idx) => (
                  <div
                    key={idx}
                    className="p-5 border border-slate-200/65 bg-white/50 hover:bg-white/80 rounded-2xl transition-all shadow-[0_4px_12px_rgba(0,0,0,0.01)]"
                  >
                    <div className="flex justify-between items-center border-b border-slate-100 pb-2.5 mb-3.5">
                      <span className="font-mono text-[10px] text-sky-600 font-black tracking-widest">{week.week.toUpperCase()} MODULE</span>
                      <span className="text-[9px] text-slate-400 font-bold tracking-wider uppercase flex items-center gap-1">
                        <BookOpen className="w-3 h-3 text-slate-400" /> Step-by-Step Task
                      </span>
                    </div>

                    <h3 className="text-sm font-black text-slate-800 tracking-tight leading-tight mb-3">
                      {week.title}
                    </h3>

                    <div className="space-y-1.5">
                      {week.topics.map((topic, topicIdx) => {
                        const stateKey = `${selectedTrack.id}-${idx}-${topicIdx}`;
                        const isDone = completedTopics[stateKey];
                        return (
                          <div
                            key={topicIdx}
                            onClick={() => toggleTopicCompletion(stateKey)}
                            className={`flex items-start gap-2.5 p-2.5 rounded-xl border text-left cursor-pointer transition-all select-none ${
                              isDone
                                ? "bg-emerald-50/50 border-emerald-100/60 text-slate-400"
                                : "bg-white border-slate-100 hover:border-slate-200 shadow-[0_2px_4px_rgba(0,0,0,0.01)] text-slate-750 text-slate-700 hover:text-slate-900"
                            }`}
                          >
                            <div className="shrink-0 mt-0.5">
                              <div className={`w-4 h-4 rounded-full border transition-all flex items-center justify-center ${
                                isDone
                                  ? "bg-emerald-500 border-emerald-500 text-white shadow-sm"
                                  : "border-slate-300 hover:border-sky-400 bg-slate-50"
                              }`}>
                                {isDone && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                              </div>
                            </div>
                            <span className={`text-[11px] font-semibold tracking-tight ${
                              isDone ? "line-through text-slate-400" : ""
                            }`}>
                              {topic}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>

            </motion.div>
          </AnimatePresence>
        </div>

        {/* ─── 🧪 Simplified Projects Arena: Liquid Glass Bento Playground ─── */}
        <div id="projects-section" className="border-t border-slate-200/80 pt-16">
          <div className="mb-14 text-center max-w-2xl mx-auto">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-50 border border-emerald-200 rounded-full text-[10px] font-mono font-bold text-emerald-700 mb-3.5 uppercase tracking-wide">
              🤖 Practical Sandboxing Hub
            </span>
            <h2 className="text-3xl md:text-5xl font-black text-slate-905 text-slate-900 tracking-tight leading-none">
              Build your own ideas.
            </h2>
            <p className="text-slate-500 text-sm md:text-base font-semibold mt-4">
              Ditch complicated terminal setups. Tweak simple variables, run predictive simulations, and inspect straightforward visual outcomes.
            </p>
          </div>

          {/* Difficulty tier tabs buttons */}
          <div className="flex items-center justify-center gap-2 mb-10 select-none">
            {(["Beginner", "Intermediate", "Advanced"] as Difficulty[]).map((dif) => {
              const isActive = activeDifficulty === dif;
              const count = PROJECTS.filter((p) => p.difficulty === dif).length;
              return (
                <button
                  key={dif}
                  onClick={() => setActiveDifficulty(dif)}
                  className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-mono font-extrabold border transition-all cursor-pointer ${
                    isActive
                      ? dif === "Beginner"
                        ? "bg-emerald-500/10 border-emerald-300 text-emerald-700 shadow-sm"
                        : dif === "Intermediate"
                        ? "bg-sky-500/10 border-sky-300 text-sky-700 shadow-sm"
                        : "bg-amber-500/10 border-amber-300 text-amber-700 shadow-sm"
                      : "bg-white/80 border-slate-200 text-slate-500 hover:bg-white hover:border-slate-300"
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full ${
                    dif === "Beginner" ? "bg-emerald-500" : dif === "Intermediate" ? "bg-sky-500" : "bg-amber-500"
                  }`} />
                  {dif} Track
                  <span className="px-1.5 py-0.5 bg-slate-100 text-slate-500 text-[9.5px] font-bold rounded-md ml-0.5">
                    {count}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            
            {/* Left projects portfolio block selector (Simplified: No arrows scroll component) */}
            <div className="lg:col-span-5 relative flex flex-col">
              <div className="flex items-center justify-between text-xs font-mono font-black text-slate-400 uppercase tracking-widest px-1 mb-4">
                <span>Select Blueprint Model</span>
                <span>Tier Total: {PROJECTS.filter((p) => p.difficulty === activeDifficulty).length}</span>
              </div>

              <div
                ref={projectsScrollRef}
                className="flex lg:flex-col gap-4 overflow-x-auto lg:overflow-x-visible pt-3 pb-4 lg:pt-0 lg:pb-0 scroll-smooth snap-x px-2"
              >
                {PROJECTS.filter((p) => p.difficulty === activeDifficulty).map((proj) => {
                  const isCurrent = selectedProject?.id === proj.id;
                  return (
                    <div
                      key={proj.id}
                      onClick={() => setSelectedProject(proj)}
                      className={`snap-start shrink-0 w-[280px] lg:w-full p-5 border rounded-2xl cursor-pointer text-left transition-all duration-300 relative ${
                        isCurrent
                          ? activeDifficulty === "Beginner"
                            ? "bg-emerald-500/10 border-emerald-400/50 shadow-[0_12px_24px_-8px_rgba(16,185,129,0.12)] -translate-y-0.5"
                            : activeDifficulty === "Intermediate"
                            ? "bg-sky-500/10 border-sky-400/50 shadow-[0_12px_24px_-8px_rgba(14,165,233,0.12)] -translate-y-0.5"
                            : "bg-amber-500/10 border-amber-400/50 shadow-[0_12px_24px_-8px_rgba(245,158,11,0.12)] -translate-y-0.5"
                          : "bg-white/40 border-white/50 hover:border-white/75 hover:bg-white/60 shadow-[0_8px_32px_0_rgba(31,38,135,0.03)]"
                      }`}
                      style={{ backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)" }}
                    >
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className={`text-[9px] font-mono font-extrabold uppercase px-2 py-0.5 rounded ${
                          activeDifficulty === "Beginner"
                            ? "bg-emerald-50 text-emerald-700 border border-emerald-100"
                            : activeDifficulty === "Intermediate"
                            ? "bg-sky-50 text-sky-700 border border-sky-100"
                            : "bg-amber-50 text-amber-700 border border-amber-100"
                        }`}>
                          Estimate: {proj.estimatedTime}
                        </span>
                        {/* No useless arrow icons as requested */}
                      </div>

                      <h3 className="font-extrabold text-sm md:text-base text-slate-800 mb-1.5 tracking-tight">
                        {proj.name}
                      </h3>

                      <p className="text-[11px] text-slate-500 leading-snug line-clamp-2">
                        {proj.whatYouWillLearn}
                      </p>

                      <div className="flex flex-wrap gap-1 mt-3.5 pt-2 border-t border-slate-100">
                        {proj.concepts.map((concept, cIdx) => (
                          <span key={cIdx} className="text-[8.5px] font-mono bg-slate-50 text-slate-550 px-1.5 py-0.5 rounded border border-slate-200/60 font-bold">
                            {concept}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right Interactive CAD Holographic Simulation Deck (SPAN 7) */}
            <div className="lg:col-span-7 flex">
              <AnimatePresence mode="wait">
                {selectedProject && (
                  <motion.div
                    key={selectedProject.id}
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.98 }}
                    transition={{ duration: 0.2 }}
                    className="bg-white/80 backdrop-blur-2xl border border-white/95 rounded-[32px] p-6 md:p-8 flex flex-col justify-between w-full shadow-[0_20px_50px_rgba(0,0,0,0.02)] relative overflow-hidden text-slate-800"
                  >
                    {/* Soft background grid pattern */}
                    <div className="absolute inset-0 bg-[radial-gradient(#00000002_1px,transparent_1px)] [background-size:20px_20px] pointer-events-none select-none" />

                    <div>
                      {/* Top bar indicator */}
                      <div className="flex items-center justify-between border-b border-slate-150 pb-4 mb-5 relative z-10">
                        <div className="flex items-center gap-2">
                          <span className="flex gap-1.5 select-none">
                            <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f56]" />
                            <span className="w-2.5 h-2.5 rounded-full bg-[#ffbd2e]" />
                            <span className="w-2.5 h-2.5 rounded-full bg-[#27c93f]" />
                          </span>
                          <span className="text-slate-400 text-[10px] font-mono ml-2 border-l border-slate-100 pl-3 uppercase">
                            lab_simulator.py
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 font-mono text-[9px] text-emerald-700 bg-emerald-50 border border-emerald-100 px-2.5 py-0.5 rounded-full font-bold select-none">
                          <span>✓ ACTIVE SIMULATOR</span>
                        </div>
                      </div>

                      {/* Details header */}
                      <div className="mb-6 relative z-10">
                        <span className="text-[8.5px] font-mono font-black text-sky-600 uppercase tracking-widest block mb-1">Target Subject</span>
                        <h2 className="text-lg md:text-xl font-black text-slate-850 tracking-tight leading-none mb-1.5">
                          {selectedProject.name}
                        </h2>
                        <p className="text-xs text-slate-500 leading-normal font-medium">
                          {selectedProject.whatYouWillLearn}
                        </p>
                      </div>

                      {/* ────────────────── Active Interactive Parameter Simulator Blocks ────────────────── */}
                      <div className="bg-slate-50/60 border border-slate-200/60 p-4 sm:p-5 rounded-2xl mb-6 relative z-10 shadow-inner">
                        <div className="flex items-center justify-between border-b border-slate-150 pb-2.5 mb-3.5">
                          <span className="text-[10px] font-mono text-slate-500 font-bold flex items-center gap-1.5 uppercase">
                            <Sliders className="w-3.5 h-3.5 text-sky-600 animate-none" /> Set Parameter Values
                          </span>
                          <span className="text-[9px] font-mono text-slate-400">Live slider simulator</span>
                        </div>

                        {/* Highly simplified, non-jargon Single Slider logic depending on chosen project */}
                        <div className="space-y-4">
                          {selectedProject.name === "Diabetes Risk Estimator" && (
                            <div>
                              <div className="flex justify-between text-xs text-slate-705 text-slate-700 mb-2 font-bold font-mono">
                                <span>Plasma Glucose Concentration</span>
                                <strong className="text-emerald-600 font-black">{sandboxInputs.glucose || 120} mg/dL</strong>
                              </div>
                              <input
                                type="range" min="40" max="300" step="1"
                                value={sandboxInputs.glucose || 120}
                                onChange={(e) => setSandboxInputs({ ...sandboxInputs, glucose: parseInt(e.target.value) })}
                                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-ew-resize accent-emerald-500"
                              />
                              <p className="text-[10px] text-slate-400 font-medium mt-2">Adjusting glucose values shifts classification hazard ranges dynamically.</p>
                            </div>
                          )}

                          {selectedProject.name === "House Prices Predictor (Kaggle Classic)" && (
                            <div>
                              <div className="flex justify-between text-xs text-slate-705 text-slate-700 mb-2 font-bold font-mono">
                                <span>Gross Interior Floor Dimension</span>
                                <strong className="text-sky-600 font-black">{sandboxInputs.sqft || 1800} Sq. Ft.</strong>
                              </div>
                              <input
                                type="range" min="500" max="4500" step="50"
                                value={sandboxInputs.sqft || 1800}
                                onChange={(e) => setSandboxInputs({ ...sandboxInputs, sqft: parseInt(e.target.value) })}
                                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-ew-resize accent-sky-500"
                              />
                              <p className="text-[10px] text-slate-400 font-medium mt-2">Predicts linear market value coordinates matching square footage size.</p>
                            </div>
                          )}

                          {selectedProject.name === "Customer Clustered Segments with K-Means" && (
                            <div>
                              <div className="flex justify-between text-xs text-slate-705 text-slate-700 mb-2 font-bold font-mono">
                                <span>Centroid Separation Segments (K Groups)</span>
                                <strong className="text-purple-600 font-black">{sandboxInputs.clusters || 4} Clusters</strong>
                              </div>
                              <input
                                type="range" min="2" max="8" step="1"
                                value={sandboxInputs.clusters || 4}
                                onChange={(e) => setSandboxInputs({ ...sandboxInputs, clusters: parseInt(e.target.value) })}
                                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-ew-resize accent-purple-500"
                              />
                              <p className="text-[10px] text-slate-400 font-medium mt-2">Splits target transaction histories into highly optimized customer groups.</p>
                            </div>
                          )}

                          {selectedProject.name === "Interactive Spambox Classifier" && (
                            <div>
                              <div className="flex justify-between text-xs text-slate-700 mb-2 font-bold font-mono">
                                <span>Custom Test Message Block</span>
                              </div>
                              <textarea
                                value={sandboxInputs.text || ""}
                                onChange={(e) => setSandboxInputs({ ...sandboxInputs, text: e.target.value })}
                                className="w-full bg-white border border-slate-200 rounded-xl p-3 text-xs font-mono focus:outline-none focus:border-sky-500 text-slate-800 focus:ring-1 focus:ring-sky-100"
                                rows={2}
                                placeholder="Try typing keywords like cash, free, winner, award..."
                              />
                              <p className="text-[10px] text-slate-400 font-medium mt-1.5">Naïve Bayes algorithms audit indicator lists inside message arrays.</p>
                            </div>
                          )}

                          {/* Default optimizer block */}
                          {selectedProject.name !== "Diabetes Risk Estimator" &&
                           selectedProject.name !== "House Prices Predictor (Kaggle Classic)" &&
                           selectedProject.name !== "Customer Clustered Segments with K-Means" &&
                           selectedProject.name !== "Interactive Spambox Classifier" && (
                            <div>
                              <div className="flex justify-between text-xs text-slate-705 text-slate-700 mb-2 font-bold font-mono">
                                <span>Model Learning Pace Step Cycles</span>
                                <strong className="text-indigo-600 font-bold font-mono">{sandboxInputs.epochs || 50} Epochs</strong>
                              </div>
                              <input
                                type="range" min="10" max="300" step="5"
                                value={sandboxInputs.epochs || 50}
                                onChange={(e) => setSandboxInputs({ ...sandboxInputs, epochs: parseInt(e.target.value) })}
                                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-ew-resize accent-indigo-500"
                              />
                              <p className="text-[10px] text-slate-400 font-medium mt-2">More cycles permit weights to calculate accurate logical mappings.</p>
                            </div>
                          )}
                        </div>

                        {/* Simulated run trigger button */}
                        <div className="mt-5">
                          <button
                            onClick={executeSandbox}
                            disabled={sandboxRunning}
                            className={`w-full py-2.5 rounded-xl text-xs font-bold transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                              sandboxRunning
                                ? "bg-slate-100 text-slate-400 border border-slate-200/80"
                                : "bg-sky-600 bg-gradient-to-b from-sky-400 to-sky-600 text-white font-extrabold hover:brightness-105 shadow-md shadow-sky-500/15"
                            }`}
                          >
                            <Play className="w-3.5 h-3.5 fill-current shrink-0" />
                            <span>{sandboxRunning ? "Training simulator models..." : "Run Simulated Training PASS"}</span>
                          </button>
                        </div>
                      </div>

                      {/* Dynamic Outputs (Cleaned up from telemetry and dense log counters) */}
                      {(sandboxOutputs || sandboxRunning || generatedTokens.length > 0) && (
                        <div className="mb-6 p-4 bg-emerald-50/50 border border-emerald-200/80 rounded-2xl relative overflow-hidden animate-fade-in text-left shadow-sm">
                          <div className="flex items-center gap-1.5 text-emerald-800 text-xs font-mono font-bold uppercase mb-3">
                            <Target className="w-3.5 h-3.5" /> Output Simulation Report
                          </div>

                          {generatedTokens.length > 0 && (
                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 text-xs text-sky-400 font-mono leading-relaxed max-h-20 overflow-y-auto mb-3 select-all">
                              {generatedTokens.map((t, idx) => (
                                <span key={idx}>{t}</span>
                              ))}
                            </div>
                          )}

                          {sandboxOutputs && (
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <span className="text-[9.5px] font-mono text-slate-400 block font-bold uppercase tracking-wider">{sandboxOutputs.metricName}</span>
                                <strong className="text-xl font-mono text-emerald-800 font-black">{sandboxOutputs.metricValue}</strong>
                              </div>
                              <div>
                                <span className="text-[9.5px] font-mono text-slate-400 block font-bold uppercase tracking-wider">Classification Cohort</span>
                                <strong className="text-xs text-slate-700 font-bold block mt-1.5">{sandboxOutputs.classification}</strong>
                              </div>
                            </div>
                          )}

                          {sandboxOutputs && (
                            <div className="mt-3.5 pt-2 border-t border-emerald-250 border-emerald-200/50 text-[10px] font-mono text-slate-500">
                              ➔ Core note: {sandboxOutputs.note}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Simplified Step Illustration cards instead of raw CLI commands/preseed modules and code editors */}
                      <div className="border-t border-slate-150 pt-5 mt-2">
                        <h4 className="text-xs font-bold text-slate-500 font-mono uppercase tracking-wider mb-4 flex items-center gap-1.5">
                          <Brain className="w-4 h-4 text-indigo-500/80" /> Beginner Blueprint Blueprint
                        </h4>
                        
                        <div className="grid grid-cols-3 gap-3 text-center">
                          <div className="bg-white/80 p-3 rounded-xl border border-slate-200/50 hover:bg-white shadow-[0_2px_8px_rgba(0,0,0,0.005)]">
                            <div className="w-7 h-7 rounded-full bg-sky-50 text-sky-600 font-mono text-xs font-black flex items-center justify-center mx-auto mb-2">1</div>
                            <span className="text-xs font-bold text-slate-800 block mb-0.5">Prepare Data</span>
                            <span className="text-[9px] text-slate-400 block leading-tight">Gather and format initial clean parameters</span>
                          </div>
                          <div className="bg-white/80 p-3 rounded-xl border border-slate-200/50 hover:bg-white shadow-[0_2px_8px_rgba(0,0,0,0.005)]">
                            <div className="w-7 h-7 rounded-full bg-indigo-50 text-indigo-600 font-mono text-xs font-black flex items-center justify-center mx-auto mb-2">2</div>
                            <span className="text-xs font-bold text-slate-800 block mb-0.5">Teach Model</span>
                            <span className="text-[9px] text-slate-400 block leading-tight">Calibrate model margins and weights</span>
                          </div>
                          <div className="bg-white/80 p-3 rounded-xl border border-slate-200/50 hover:bg-white shadow-[0_2px_8px_rgba(0,0,0,0.005)]">
                            <div className="w-7 h-7 rounded-full bg-emerald-50 text-emerald-600 font-mono text-xs font-black flex items-center justify-center mx-auto mb-2">3</div>
                            <span className="text-xs font-bold text-slate-800 block mb-0.5">Solve / Output</span>
                            <span className="text-[9px] text-slate-400 block leading-tight">Validate outcomes instantly on request</span>
                          </div>
                        </div>
                      </div>

                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}
