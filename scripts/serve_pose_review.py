"""Loopback-only static review preview, including byte ranges for MP4 seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range=None
        requested=self.headers.get('Range')
        path=Path(self.translate_path(self.path))
        if not requested or not path.is_file():return super().send_head()
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
        size=path.stat().st_size
        try:
            if not match or not any(match.groups()) or size==0:raise ValueError()
            a,b=match.groups()
            if a:
                start=int(a);end=min(int(b),size-1) if b else size-1
            else:
                start=max(0,size-int(b));end=size-1
            if not 0<=start<=end<size:raise ValueError()
        except ValueError:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return None
        stream=path.open('rb');stream.seek(start);self.byte_range=(start,end)
        self.send_response(206);self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(end-start+1));self.end_headers();return stream

    def copyfile(self,source,outputfile):
        if self.byte_range is None:return super().copyfile(source,outputfile)
        left=self.byte_range[1]-self.byte_range[0]+1
        try:
            while left:
                data=source.read(min(65536,left))
                if not data:break
                outputfile.write(data);left-=len(data)
        except (BrokenPipeError,ConnectionResetError):pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/yolo-comparison'))
    p.add_argument('--port',type=int,default=18591);args=p.parse_args()
    print(f'http://127.0.0.1:{args.port}/review/index.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(args.root.resolve()))).serve_forever()


if __name__=='__main__':main()
