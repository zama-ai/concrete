"""Extract some info about the host machine."""
import json
import os
import platform
import urllib.parse

import cpuinfo
import dotenv
import psutil


def main():
    """Extract some info about the host machine."""
    dotenv.load_dotenv()

    properties = []

    cpu_value = cpuinfo.get_cpu_info()["brand_raw"].replace("(R)", "®").replace("(TM)", "™")
    properties.append(["CPU", cpu_value])

    vcpu_value = os.getenv("VCPU")
    if vcpu_value is not None:
        properties.append(["vCPU", vcpu_value])

    ram_value = f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"
    properties.append(["RAM", ram_value])

    os_value = os.getenv("OS_NAME")
    if os_value is None:
        os_value = f"{platform.system()} {platform.release()}"
    properties.append(["OS", os_value])

    name = os.getenv("MACHINE_NAME").strip()
    if name is None:
        name = platform.node().strip()

    id_ = name.lower()
    id_ = id_.replace(" ", "-")
    id_ = id_.replace("_", "-")
    id_ = id_.replace(".", "-")
    id_ = id_.replace("(", "")
    id_ = id_.replace(")", "")
    id_ = id_.replace("$/h", "-dollars-per-hour")
    id_ = id_.strip()
    id_ = urllib.parse.quote_plus(id_)

    machine = {"id": id_, "name": name, "properties": properties}
    with open(".benchmarks/machine.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
