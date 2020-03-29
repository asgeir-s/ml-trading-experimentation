import { URLSearchParams } from "url"

const fetch = require("node-fetch")
const BASE_ENDPOINT = "https://api.binance.com"
const DATA_FILE_PATH = "data/"

const readLastLines = require("read-last-lines")

/**
 * symbol 	STRING 	YES
 * fromId 	LONG 	NO 	id to get aggregate trades from INCLUSIVE.
 * startTime 	LONG 	NO 	Timestamp in ms to get aggregate trades from INCLUSIVE.
 * endTime 	LONG 	NO 	Timestamp in ms to get aggregate trades until INCLUSIVE.
 * limit 	INT 	NO 	Default 500; max 1000.
 */

export async function getCandlesticks(
  symbol: string,
  interval: string,
  startTime?: number,
  endTime?: number,
  limit: number = 1000
) {
  const params = new URLSearchParams({
    symbol,
    interval,
    ...(startTime != null ? { fromId: startTime.toString() } : {}),
    ...(endTime != null ? { endTime: endTime.toString() } : {}),
    limit: limit.toString(),
  })
  const url = BASE_ENDPOINT + "/api/v3/klines?" + params
  const res = await fetch(url)

  if (res.ok) {
    return res.json()
  } else {
    throw {
      name: res.statusText,
      codePath: "data-loading/binance/candlestick-to-csv.ts",
      url,
      message: await res.json(),
    }
  }
}

async function getLastCandlestickTime(fileName) {
  try {
    readLastLines
      .read(DATA_FILE_PATH + fileName, 1)
      .then(lines => console.log(lines))
  } catch (e) {
    return 0
  }
}

function candlestickToCSV(symbol: string, interval: string) {
  const fileName = "candlestick-" + symbol + "-" + interval + ".csv"

  const lastTime = getLastCandlestickTime(fileName)

  // get candlestick data 
}
