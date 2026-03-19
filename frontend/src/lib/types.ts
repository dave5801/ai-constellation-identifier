export type PossiblePlanet = {
  x: number;
  y: number;
  brightness: number;
};

export type ConstellationMatch = {
  name: string;
  confidence: number;
};

export type IdentifyResponse = {
  constellations: ConstellationMatch[];
  stars_detected: number;
  possible_planets: PossiblePlanet[];
  annotated_image: string;
};
