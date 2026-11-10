#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Error: enter the interface name as arg1 (and port to open arg2)"
    exit 1
fi

iface=$1
port=${2:-8080}

sudo ip link set fctap0 down
sudo ip tuntap del dev fctap0 mode tap

sudo ip tuntap add dev fctap0 mode tap
sudo ip addr add 172.16.0.1/24 dev fctap0
sudo ip link set fctap0 up

sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o "$iface" -j MASQUERADE
sudo iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i fctap0 -o "$iface" -j ACCEPT
sudo iptables -A INPUT -i fctap0 -p tcp --dport "$port" -j ACCEPT

ip a
