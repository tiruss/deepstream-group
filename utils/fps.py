################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
"""성능 모니터링을 위한 FPS 측정 클래스."""

import time

start_time = time.time()
frame_count = 0


class FPS:
    """성능 모니터링을 위한 FPS 측정 클래스."""

    def __init__(self, stream_id: int) -> None:
        global start_time
        self.start_time = start_time
        self.is_first = True
        global frame_count
        self.frame_count = frame_count
        self.stream_id = stream_id

    def __call__(self) -> None:
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        if (end_time - self.start_time) > 5:
            print("*" * 30)
            print(
                f"FPS of stream {self.stream_id} is {float(self.frame_count) / 5.0:.2f}",
                flush=True,
            )
            self.frame_count = 0
            self.start_time = end_time
        else:
            self.frame_count += 1
