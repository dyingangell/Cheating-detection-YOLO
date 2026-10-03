from multiprocessing import shared_memory
import os
import time

SHM_NAME = "cv_frame_buffer"
# Configurable via env vars to avoid always allocating 527 MB.
# Example for 8 cameras at 720p: SHM_FRAMES=50
_frames = int(os.getenv("SHM_FRAMES", "200"))
_h = int(os.getenv("SHM_HEIGHT", "720"))
_w = int(os.getenv("SHM_WIDTH", "1280"))
SIZE = _frames * _h * _w * 3

def start_master():
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SIZE)
        print(f"Shared Memory '{SHM_NAME}' created and held")
        print("Don't close this terminal window until workers are done!")

        # Infinite loop to keep the process alive
        while True:
            time.sleep(10)

    except FileExistsError:
        print("Shared Memory already exists.")
    except KeyboardInterrupt:
        print("\nDeleting Shared Memory and exiting...")
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    start_master()