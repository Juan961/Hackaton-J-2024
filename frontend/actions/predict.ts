import axios from 'axios';

interface IDataClassificationImage {
  Sunlight_Hours: number
  Temperature: number
  Humidity: number
  Soil_Type: string // "sandy" | "loam" | "clay"
  Water_Frequency: string // "daily" | "weekly" | "monthly"
  Fertilizer_Type: string // "organic" | "chemical" | "none"
}

interface IDataImage {
  image: string;
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
