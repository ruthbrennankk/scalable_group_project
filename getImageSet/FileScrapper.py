import asyncio
import time
import os
import argparse
import aiohttp

async def download_pep(myfilename,session,sem) -> None:
     
    url = f"https://cs7ns1.scss.tcd.ie/index.php?download=noresume_speed&shortname=joglekac&myfilename={myfilename}"
    outputfilename = f"trainingData/{myfilename}"
    async with sem:
        print(f"Begin downloading {url}")
        async with session.get(url) as res:
            content = await res.read()
        if res.status != 200:
            print(f"Download failed: {res.status}")
            return
        await write_to_file(myfilename,content)

async def write_to_file(myfilename: str, content: bytes) -> None:
    filename = f"trainingData/{myfilename}"
    with open(filename, "wb") as pep_file:
        print(f"Begin writing to {myfilename}")
        pep_file.write(content)
        print(f"Finished writing {myfilename}")


# async def web_scrape_task(myfilename: str) -> None:
#     content = await download_pep(myfilename)
#     await write_to_file(myfilename, content)


async def main(filelist):
    tasks = []
    sem = asyncio.Semaphore(7)
    async with aiohttp.ClientSession() as client_session:
        for furl in filelist:
            tasks.append(download_pep(furl,client_session,sem))
        await asyncio.wait(tasks)

if __name__ == '__main__':
    filelist = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFileName', type=str)
    args = parser.parse_args()
    with open(args.inputFileName,'r') as f:
        for line in f:
            name = line.strip()
            filelist.append(name[:-1])
    asyncio.run(main(filelist))
