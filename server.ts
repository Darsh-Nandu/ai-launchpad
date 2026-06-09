import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware to parse JSON bodies
  app.use(express.json());

  // Lazy-loaded Gemini SDK client
  let ai: GoogleGenAI | null = null;
  function getAI(): GoogleGenAI {
    if (!ai) {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
        throw new Error(
          "GEMINI_API_KEY is not configured. Please set your Gemini API Key in the Secrets panel."
        );
      }
      ai = new GoogleGenAI({ apiKey });
    }
    return ai;
  }

  // API route for doubt solving
  app.post("/api/claude/generate", async (req, res) => {
    try {
      const { prompt } = req.body;
      if (!prompt) {
        return res.status(400).json({ error: "The prompt query cannot be empty." });
      }

      const client = getAI();
      const response = await client.models.generateContent({
        model: "gemini-2.0-flash",
        contents: prompt,
        config: {
          systemInstruction:
            "You are an inspiring, extremely knowledgeable AI/ML mentor and researcher named Darsh Nandu. " +
            "Your job is to clarify complex concepts in Machine Learning, Deep Learning, Generative AI, or Math. " +
            "Provide elegant, clear, structured explanations with markdown formatting (use bullet points, bold keywords, " +
            "and short examples). Always adopt an encouraging, professional, and friendly academic tone.",
        },
      });

      res.json({ text: response.text });
    } catch (error: any) {
      console.error("AI server route error:", error);
      res.status(500).json({
        error: error.message || "An unexpected error occurred inside the AI server.",
      });
    }
  });

  // Setup Vite middleware for Asset Serving & HMR-like compilation
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
});
