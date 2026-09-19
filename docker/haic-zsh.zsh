export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"

if [[ "${VIRTUAL_ENV:-}" != "/venv/main" && -f /venv/main/bin/activate ]]; then
    source /venv/main/bin/activate
fi

_haic_sync_tmux_env() {
    [[ -n "${TMUX:-}" ]] || return 0

    local name statement
    for name in DISPLAY XAUTHORITY SSH_AUTH_SOCK; do
        statement="$(tmux show-environment -s "$name" 2>/dev/null)" || continue
        eval "$statement"
    done
}

if [[ -n "${TMUX:-}" ]]; then
    autoload -Uz add-zsh-hook
    add-zsh-hook precmd _haic_sync_tmux_env
    _haic_sync_tmux_env
fi

clip() {
    if (( $# != 1 )); then
        print -u2 "usage: clip <file>"
        return 2
    fi

    _haic_sync_tmux_env

    if [[ -z "${DISPLAY:-}" ]]; then
        print -u2 "DISPLAY is empty. Connect with X11 forwarding, for example: ssh -X ..."
        return 1
    fi

    xclip -selection clipboard < "$1"
}

tmuxreload() {
    tmux source-file /etc/tmux.conf
    [[ -f "$HOME/.tmux.conf" ]] && tmux source-file "$HOME/.tmux.conf"
}

alias ll='ls -lah --color=auto'
alias gs='git status --short --branch'
alias gl='git log --oneline --decorate -20'
