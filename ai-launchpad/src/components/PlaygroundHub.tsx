import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Send, AlertCircle, RefreshCw, GraduationCap, BookOpen, Heart, Award, Cpu, Terminal, User, Key } from "lucide-react";
import { GoogleGenAI } from "@google/genai";
import { ChatMessage } from "../types";

interface PlaygroundHubProps {
  onNavigateToStudio: () => void;
  onNavigateToLibrary: () => void;
}

export default function PlaygroundHub({
  onNavigateToStudio,
  onNavigateToLibrary,
}: PlaygroundHubProps) {
  const [chatInput, setChatInput] = useState("");
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("gemini_api_key") || "");
  const [chatLog, setChatLog] = useState<ChatMessage[]>([
    {
      id: "welcome-msg",
      sender: "system",
      text: "Hello neural explorer! I am your AI Mentor, Darsh Nandu. Ask me any technical doubts about algorithms, tensors, linear algebra, backpropagation, or prompt engineering!",
      timestamp: new Date(),
    },
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const endOfChatRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    endOfChatRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Only auto-scroll down if there are active interactions (more than 1 message, or currently generating)
    // This ensures the page defaults cleanly to the header section on the first load instead of autoscrolling down
    if (chatLog.length > 1 || isGenerating) {
      scrollToBottom();
    }
  }, [chatLog, isGenerating]);

  // Handler for AI doubt solver
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const query = chatInput.trim();
    if (!query || isGenerating) return;

    setErrorMessage(null);
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: "user",
      text: query,
      timestamp: new Date(),
    };

    setChatLog((prev) => [...prev, userMsg]);
    setChatInput("");
    setIsGenerating(true);

    try {
      if (!apiKey) throw new Error("Please enter your Gemini API key above to use the AI mentor.");

      const ai = new GoogleGenAI({ apiKey });
      const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: query,
        config: {
          systemInstruction:
            "You are an inspiring, extremely knowledgeable AI/ML mentor and researcher named Darsh Nandu. " +
            "Your job is to clarify complex concepts in Machine Learning, Deep Learning, Generative AI, or Math. " +
            "Provide elegant, clear, structured explanations with markdown formatting (use bullet points, bold keywords, " +
            "and short examples). Always adopt an encouraging, professional, and friendly academic tone.",
        },
      });

      const claudeMsg: ChatMessage = {
        id: `claude-${Date.now()}`,
        sender: "claude",
        text: response.text || "I processed your request, but returned an empty response.",
        timestamp: new Date(),
      };

      setChatLog((prev) => [...prev, claudeMsg]);
    } catch (error: any) {
      console.error(error);
      setErrorMessage(error.message || "Could not reach the AI assistant.");
      setChatLog((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          sender: "system",
          text: `⚠️ ERROR: ${error.message || "An unexpected error occurred."}`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  // Clears chat log back to initial greeting
  const handleClearChat = () => {
    setChatLog([
      {
        id: "welcome-msg-reset",
        sender: "system",
        text: "Reset complete! I'm ready for your next set of AI/ML doubts. Ask away!",
        timestamp: new Date(),
      },
    ]);
    setErrorMessage(null);
  };

  // Formatter to render response text nicely (bold, points) safely
  const formatResponseText = (text: string) => {
    // Process markdown codeblocks or inline backticks
    return text.split("\n").map((line, idx) => {
      let trimmed = line.trim();

      // Bold titles or headers
      if (trimmed.startsWith("###")) {
        return (
          <h4 key={idx} className="font-bold text-gray-900 text-lg mt-3 mb-1">
            {trimmed.replace("###", "").trim()}
          </h4>
        );
      }
      if (trimmed.startsWith("##")) {
        return (
          <h3 key={idx} className="font-bold text-gray-900 text-xl mt-4 mb-2 border-b border-gray-200 pb-1">
            {trimmed.replace("##", "").trim()}
          </h3>
        );
      }
      if (trimmed.startsWith("#")) {
        return (
          <h2 key={idx} className="font-semibold text-gray-900 text-2xl mt-4 mb-2">
            {trimmed.replace("#", "").trim()}
          </h2>
        );
      }

      // Bullets
      if (trimmed.startsWith("*") || trimmed.startsWith("-")) {
        const itemText = trimmed.substring(1).trim();
        return (
          <ul key={idx} className="list-disc pl-6 py-0.5 text-gray-800">
            <li>{renderInlineFormatting(itemText)}</li>
          </ul>
        );
      }

      // Standard paragraphs
      return (
        <p key={idx} className="leading-relaxed text-gray-800 my-1.5">
          {renderInlineFormatting(line)}
        </p>
      );
    });
  };

  // Inline formatter for bold (**text**) or code (`code`)
  const renderInlineFormatting = (line: string) => {
    // Replace **text** logic with JSX parts
    const parts = line.split(/(\*\*.*?\*\*|`.*?`)/);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return (
          <strong key={i} className="font-bold text-black bg-yellow-100 px-1 rounded-sm">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return (
          <code key={i} className="bg-gray-100 text-pink-600 px-1.5 py-0.5 rounded text-sm font-mono border border-gray-200">
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });
  };

  return (
    <div id="playground-hub" className="min-h-screen bg-[#ece9e6] bg-radial from-[#ffffff] to-[#dfdad6] text-black pt-16 pb-20 relative overflow-hidden">
      
      {/* Dynamic Flying shapes in background - Flying Papers cartoon style */}
      <div className="hidden md:flex absolute top-20 left-[-5%] w-48 h-48 bg-[#9fc5e8] opacity-30 border-2 border-black rounded-full select-none pointer-events-none transform -rotate-12 items-center justify-center shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] text-[#1a56db]">
        <Cpu className="w-16 h-16 stroke-[1.5]" />
      </div>
      <div className="hidden md:flex absolute bottom-20 right-[-5%] w-52 h-52 bg-[#f9cb9c] opacity-30 border-2 border-black rounded-3xl select-none pointer-events-none transform rotate-12 items-center justify-center shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] text-[#e56845]">
        <Terminal className="w-16 h-16 stroke-[1.5]" />
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 sm:mt-12 relative z-10">
        
        {/* RETRO BRAND HERO SECTION */}
        <div id="retro-hero" className="bg-[#b4c7e7] border-4 border-black p-5 sm:p-8 md:p-12 rounded-3xl shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] sm:shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] text-center mb-12 sm:mb-16 transform -rotate-1 md:rotate-1 hover:rotate-0 transition-transform duration-300">
          
          <div className="inline-flex items-center gap-2 bg-[#ffe599] border-2 border-black px-4 py-1.5 rounded-full text-xs sm:text-sm font-bold uppercase mb-6 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]">
            <span>✨ Sparking ML Minds</span>
          </div>

          <h1 className="font-extrabold text-2xl sm:text-4xl md:text-6xl text-black leading-tight tracking-tight mb-6">
            Your Complete AI Learning Journey, <br className="hidden sm:inline"/>
            <span className="bg-[#f3ca52] text-black px-4 py-1.5 sm:py-0.5 border-2 border-black inline-block mt-2 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] sm:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
              Starts Here.
            </span>
          </h1>

          <p className="max-w-3xl mx-auto font-medium text-sm sm:text-lg md:text-xl text-gray-900 leading-relaxed mb-8">
            No more scattered resources. No more confusion. Everything you 
            need to go from zero to AI engineer — curated by Darsh Nandu.
          </p>

          {/* CTA Buttons in Retro style */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-10">
            <button
              onClick={onNavigateToStudio}
              id="start-learning-btn"
              className="w-full sm:w-auto bg-[#3d85c6] hover:bg-[#29689e] text-white font-extrabold text-base sm:text-lg px-6 sm:px-8 py-3.5 sm:py-4 rounded-xl border-2 border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] sm:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] active:translate-x-[3px] active:translate-y-[3px] transition-all duration-150 cursor-pointer"
            >
              Start Learning →
            </button>
            <button
              onClick={onNavigateToLibrary}
              id="watch-resources-btn"
              className="w-full sm:w-auto bg-[#ffd966] hover:bg-[#f1c232] text-black font-extrabold text-base sm:text-lg px-6 sm:px-8 py-3.5 sm:py-4 rounded-xl border-2 border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] sm:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] active:translate-x-[3px] active:translate-y-[3px] transition-all duration-150 cursor-pointer animate-pulse"
            >
              Watch Resources
            </button>
          </div>

          {/* Stats Row */}
          <div className="border-t-2 border-black pt-6 flex flex-wrap items-center justify-center gap-y-4 gap-x-6 sm:gap-x-8 text-xs sm:text-sm md:text-base font-bold text-gray-900">
            <div className="flex items-center gap-2">
              <span className="bg-[#ffe599] w-7 h-7 border-2 border-black rounded-full flex items-center justify-center shadow-[1px_1px_0px_0px_rgba(0,0,0,1)]">
                <GraduationCap className="w-4 h-4 text-black" />
              </span>
              <span>10+ Learning Paths</span>
            </div>
            <div className="hidden md:block text-black">|</div>
            <div className="flex items-center gap-2">
              <span className="bg-[#c9daf8] w-7 h-7 border-2 border-black rounded-full flex items-center justify-center shadow-[1px_1px_0px_0px_rgba(0,0,0,1)]">
                <BookOpen className="w-4 h-4 text-black" />
              </span>
              <span>50+ Curated Resources</span>
            </div>
            <div className="hidden md:block text-black">|</div>
            <div className="flex items-center gap-2">
              <span className="bg-[#d9ead3] w-7 h-7 border-2 border-black rounded-full flex items-center justify-center shadow-[1px_1px_0px_0px_rgba(0,0,0,1)]">
                <Heart className="w-4 h-4 text-black fill-black/15" />
              </span>
              <span>100% Free</span>
            </div>
            <div className="hidden md:block text-black">|</div>
            <div className="flex items-center gap-2">
              <span className="bg-[#e06666] w-7 h-7 border-2 border-black rounded-full flex items-center justify-center shadow-[1px_1px_0px_0px_rgba(0,0,0,1)]">
                <Award className="w-4 h-4 text-black" />
              </span>
              <a
                href="https://www.linkedin.com/in/darsh-nandu-072a39373"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-[#3d85c6] transition-colors cursor-pointer"
              >
                Built by Darsh Nandu
              </a>
            </div>
          </div>
        </div>

        {/* AI DOUBT SOLVER CHAT CONSOLE */}
        <div id="ai-doubt-solver-section" className="bg-white border-4 border-black rounded-2xl sm:rounded-3xl shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] sm:shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] overflow-hidden">
          
          {/* Chat Header in Flying Papers bold theme */}
          <div className="bg-[#ffe599] border-b-4 border-black px-4 sm:px-6 py-3 sm:py-4 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="bg-[#3d85c6] w-9 h-9 sm:w-10 sm:h-10 border-2 border-black rounded-xl flex items-center justify-center shadow-[1.5px_1.5px_0px_0px_rgba(0,0,0,1)] sm:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] text-[#ffe599] shrink-0">
                <Cpu className="w-4 h-4 sm:w-5 sm:h-5" />
              </div>
              <div>
                <h2 className="font-extrabold text-base sm:text-xl text-black">
                  Ask Anything About AI & ML
                </h2>
                <p className="text-[10px] sm:text-xs font-semibold text-gray-700 uppercase tracking-wider">
                  Powered by Claude AI
                </p>
              </div>
            </div>

            <button
              onClick={handleClearChat}
              className="bg-[#f4cccc] hover:bg-[#ea9999] border-2 border-black px-3 py-1 rounded-lg text-xs font-bold transition-colors cursor-pointer flex items-center gap-1 shadow-[1.5px_1.5px_0px_0px_rgba(0,0,0,1)] sm:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-none active:translate-x-0.5 active:translate-y-0.5"
            >
              Reset Chat
            </button>
          </div>

          {/* Chat Logs Canvas */}
          {/* API Key Input Strip */}
          <div className="bg-[#fff9e6] border-b-2 border-black px-4 py-2 flex items-center gap-2">
            <Key className="w-4 h-4 text-gray-500 shrink-0" />
            <input
              type="password"
              placeholder="Enter your Gemini API key to activate the AI mentor…"
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                localStorage.setItem("gemini_api_key", e.target.value);
              }}
              className="flex-1 text-xs font-mono bg-transparent outline-none text-gray-700 placeholder-gray-400"
            />
            {apiKey && (
              <span className="text-[10px] font-bold text-green-600 uppercase tracking-wider">✓ Key saved</span>
            )}
          </div>

          {/* Chat Logs Canvas */}
          <div className="h-[360px] sm:h-[400px] overflow-y-auto p-3 sm:p-6 bg-[#f9f9f9] flex flex-col gap-4 border-b-2 border-gray-200">
            <AnimatePresence>
              {chatLog.map((msg) => (
                <motion.div
                 key={msg.id}
                 initial={{ opacity: 0, y: 10 }}
                 animate={{ opacity: 1, y: 0 }}
                 className={`flex flex-col max-w-[92%] sm:max-w-[85%] ${
                   msg.sender === "user" ? "self-end ml-auto" : "self-start mr-auto"
                 }`}
                >
                  <span className={`text-[9px] sm:text-[10px] font-mono mb-1 text-gray-400 font-bold uppercase flex items-center gap-1 ${
                    msg.sender === "user" ? "justify-end text-right" : "justify-start text-left"
                  }`}>
                    {msg.sender === "user" ? (
                      <>
                        <User className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-gray-500" />
                        <span>Student Learner</span>
                      </>
                    ) : msg.sender === "claude" ? (
                      <>
                        <Cpu className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-[#3d85c6]" />
                        <span>Darsh Nandu (AI Mentor)</span>
                      </>
                    ) : (
                      <>
                        <Terminal className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-emerald-600" />
                        <span>AI Engine Client</span>
                      </>
                    )}
                  </span>

                  <div
                    className={`p-3 sm:p-4 rounded-xl sm:rounded-2xl border-2 border-black font-medium leading-relaxed text-xs sm:text-sm md:text-base shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] sm:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] ${
                      msg.sender === "user"
                        ? "bg-[#cfe2f3] text-black rounded-tr-none"
                        : msg.sender === "claude"
                        ? "bg-white text-black rounded-tl-none"
                        : "bg-[#d9ead3] text-black rounded-tl-none"
                    }`}
                  >
                    {msg.sender === "claude" ? (
                      <div className="space-y-1">{formatResponseText(msg.text)}</div>
                    ) : (
                      <p>{msg.text}</p>
                    )}
                  </div>
                </motion.div>
              ))}

              {isGenerating && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="self-start mr-auto flex items-center gap-2 bg-[#fff2cc] border-2 border-black p-2.5 sm:p-3 rounded-xl sm:rounded-2xl shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] sm:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] font-bold text-xs sm:text-sm"
                >
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#e69138]" />
                  <span>Synthesizing mathematical response...</span>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={endOfChatRef} />
          </div>

          {/* Input Box Controls */}
          <form onSubmit={handleSendMessage} className="p-3 sm:p-4 bg-white flex items-center gap-2 sm:gap-3">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask me calculations, algorithms, parameters or doubts..."
              disabled={isGenerating}
              className="flex-1 bg-gray-50 border-2 border-black p-2.5 sm:p-3.5 rounded-xl font-bold text-black text-xs sm:text-sm focus:outline-none focus:bg-white placeholder-gray-400"
            />
            <button
              type="submit"
              disabled={isGenerating || !chatInput.trim()}
              className="bg-[#3d85c6] hover:bg-[#29689e] text-white p-2.5 sm:p-3.5 rounded-xl border-2 border-black transition-transform active:scale-95 disabled:opacity-55 disabled:scale-100 disabled:pointer-events-none shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] cursor-pointer shrink-0"
            >
              <Send className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>
          </form>

        </div>

        {/* ERROR DISPATCH PANEL */}
        {errorMessage && (
          <div className="mt-4 bg-[#f4cccc] border-2 border-black p-4 rounded-xl flex items-center gap-3 shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]">
            <AlertCircle className="w-6 h-6 text-red-700 flex-shrink-0" />
            <p className="text-sm font-bold text-red-900">
              {errorMessage}
            </p>
          </div>
        )}

      </div>
    </div>
  );
}
