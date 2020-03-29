import * as readLastLines from "read-last-lines"
import * as fetch from "node-fetch"
import { promises as fsP } from "fs"
import * as fs from "fs"
import * as R from "ramda"
import * as csvWriter from "csv-write-stream"
import { URLSearchParams } from "url"

const BASE_ENDPOINT = "https://api.binance.com"
const DATA_FILE_PATH = "binance/data/"

/**
 * symbol 	STRING 	YES
 * fromId 	LONG 	NO 	id to get aggregate trades from INCLUSIVE.
 * startTime 	LONG 	NO 	Timestamp in ms to get aggregate trades from INCLUSIVE.
 * endTime 	LONG 	NO 	Timestamp in ms to get aggregate trades until INCLUSIVE.
 * limit 	INT 	NO 	Default 500; max 1000.
 *
 * returns:
 * [
 *  [
 *   1499040000000,      // Open time
 *   "0.01634790",       // Open
 *   "0.80000000",       // High
 *   "0.01575800",       // Low
 *   "0.01577100",       // Close
 *   "148976.11427815",  // Volume
 *   1499644799999,      // Close time
 *   "2434.19055334",    // Quote asset volume
 *   308,                // Number of trades
 *   "1756.87402397",    // Taker buy base asset volume
 *   "28.46694368",      // Taker buy quote asset volume
 *   "17928899.62484339" // Ignore.
 *  ]
 * ]
 */

export async function getCandlesticks(
  symbol: string,
  interval: string,
  startTime?: number,
  endTime?: number,
  limit: number = 1000
): Promise<[][]> {
  const params = new URLSearchParams({
    symbol,
    interval,
    ...(startTime != null ? { startTime: startTime.toString() } : {}),
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

async function getLastCandlestickTime(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      return readLastLines
        .read(filePath, 1)
        .then((lines: string) => lines.split(",")[6])
    } else {
      return 0
    }
  } catch (e) {
    return 0
  }
}

async function candlestickToCSV(symbol: string, interval: string) {
  const fileName = "candlestick-" + symbol + "-" + interval + ".csv"
  const filePath = DATA_FILE_PATH + fileName
  let newDataPoints = 0

  const writer = csvWriter({
    headers: [
      "open time",
      "open",
      "high",
      "low",
      "close",
      "volume",
      "close time",
      "quote asset volume",
      "number of trades",
      "taker buy base asset volume",
      "taker buy quote asset volume",
      "ignore",
    ],
    sendHeaders: !fs.existsSync(filePath),
  })

  let lastTime = await getLastCandlestickTime(filePath)
  if (!fs.existsSync(filePath)) {
    console.log("The file does not exist. Will create file.")
    try {
      await fsP.writeFile(filePath, "")
      console.log("File created at " + filePath)
    } catch (e) {
      console.log("Error, could not create file!")
      throw e
    }
  }

  writer.pipe(fs.createWriteStream(filePath, { flags: "a" }))

  let newDataPointsInThisRequest = 1
  while (newDataPointsInThisRequest > 0) {
    console.log("last line: " + lastTime)
    const candlesticks = await getCandlesticks(symbol, interval, lastTime)

    newDataPointsInThisRequest = candlesticks.length
    newDataPoints += candlesticks.length
    R.map(data => {
      writer.write(data)
    }, candlesticks)
    lastTime = await getLastCandlestickTime(filePath)
  }
  writer.end()

  console.log("Number of new data points: " + newDataPoints)
}

candlestickToCSV("BTCUSDT", "1h")
