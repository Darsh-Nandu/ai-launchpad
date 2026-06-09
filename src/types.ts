export enum HubType {
  PLAYGROUND = "PLAYGROUND",
  STUDIO = "STUDIO",
  LIBRARY = "LIBRARY",
}

export type Difficulty = "Beginner" | "Intermediate" | "Advanced";

export interface RoadmapTrack {
  id: string;
  iconName: string;
  title: string;
  description: string;
  difficulty: Difficulty;
  duration: string;
  syllabus: {
    week: string;
    title: string;
    topics: string[];
  }[];
}

export interface ResourceItem {
  id: string;
  name: string;
  description: string;
  tags: string[];
  url: string;
}

export interface VideoItem {
  id: string;
  youtubeId: string;
  title: string;
  channel: string;
  badge: string; // "Playlist" or duration e.g. "45 min"
  playlistId?: string;
}

export interface ProjectItem {
  id: string;
  name: string;
  whatYouWillLearn: string;
  concepts: string[];
  estimatedTime: string;
  difficulty: Difficulty;
}

export interface ChatMessage {
  id: string;
  sender: "user" | "claude" | "system";
  text: string;
  timestamp: Date;
}
