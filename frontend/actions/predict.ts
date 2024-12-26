import axios from 'axios';

interface IDataClassificationImage {
  sunlightHours: number
  temperature: number
  humidity: number
  soilType: string // "sandy" | "loam" | "clay"
  waterFrequency: string // "daily" | "weekly" | "monthly"
  fertilizerType: string // "organic" | "chemical" | "none"
}

interface IDataImage {
  base64: string;
}

export interface IResponseClassification {
  growing: boolean
  response: string
}

export interface IResponseImage {
  response: string
  plant: string
}

export const predict = async (accessToken: string, model:"image"|"classification", input:IDataClassificationImage|IDataImage): Promise<IResponseClassification|IResponseImage|null> => {
  try {
    const config = useRuntimeConfig();

    const url = config.public.apiUrl + '/predict/' + model;

    const { data } = await axios.post(url, input, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': accessToken
      }
    });

    return data;

  } catch (error) {
    console.error(error);

    return null;
  }
};
