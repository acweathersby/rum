FROM rust:1.81.0-bullseye as baseline

RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

RUN apt-get update

RUN apt-get -y install zsh

RUN rustup default nightly

RUN curl -LJOo hugo.deb .deb https://github.com/gohugoio/hugo/releases/download/v0.117.0/hugo_extended_0.117.0_linux-amd64.deb && dpkg -i hugo.deb

RUN hugo version

CMD [ "zsh" ]