FROM debian:bookworm-slim

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG HOME=/home/$USERNAME

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates sudo curl zsh git  \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd -g $USER_GID $USERNAME \
    && useradd -u $USER_UID -g $USERNAME -d $HOME -s /bin/zsh -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME 

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d

USER $USERNAME

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && sed -i 's/^ZSH_THEME=.*/ZSH_THEME="theunraveler"/' $HOME/.zshrc