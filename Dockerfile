FROM alpine as build
RUN apk add git gcc g++ make cmake && \
    git clone --recursive https://github.com/li-plus/chatglm.cpp.git && \
    cd chatglm.cpp  && \
    git submodule update --init --recursive && \
    cmake -B build && cmake --build build -j

FROM alpine:latest
COPY --from=build /chatglm.cpp/build/bin/main /chatglm
RUN apk add libc-dev libstdc++-dev &&  apk cache clean
