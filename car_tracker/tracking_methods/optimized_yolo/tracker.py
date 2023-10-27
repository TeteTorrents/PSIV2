import math

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.whdict = {}
        self.up = 0
        self.down = 0
        self.counted_id = []

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + w) // 2
            cy = (y + h) // 2

            same_object_detected = False
            try:
                for id, ptl in self.center_points.items():
                    pt = ptl[-1]
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 100 and len(self.center_points[id]) < 3: # < 3 UwU
                        self.center_points[id].append((cx, cy))
                        self.whdict[id] = (w, h)
                        objects_bbs_ids.append([x, y, w, h, id])
                        same_object_detected = True
                        break
                    
                    if len(self.center_points[id]) >= 3:
                        same_object_detected = True
                        objects_bbs_ids.append(self.self_update(id))
            except:
                pass                

            if same_object_detected is False:
                self.center_points[self.id_count] = [(cx, cy)]
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        try:
            for id in self.center_points:
                if len(self.center_points[id]) >= 3:
                    objects_bbs_ids.append(self.self_update(id))
        except:
            pass

        return objects_bbs_ids

    def self_update(self, id):
        print("YOP")
        coords = self.center_points[id]
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]

        direction = "up" if y[-1] < y[0] else "down"
        velocity = (y[-1] - y[-3]) / 2 # len(y) - 1

        if direction == "up":
            predicted_x = x[-1]
            predicted_y = max(y[-1] + velocity, 223) 
        elif direction == "down":
            predicted_x = x[-1]
            predicted_y = min(y[-1] + velocity, 509)
        
        self.center_points[id].append((predicted_x, predicted_y))

        if (predicted_y == 223 and direction == 'up') or (predicted_y == 509 or direction == 'down'):
            del self.center_points[id]
        
        if int(predicted_y) in [i for i in range(250, 435)] and id not in self.counted_id:
            if direction == 'up':
                self.up += 1
            else:
                self.down += 1
            self.counted_id.append(id)

        return [predicted_x, predicted_y, self.whdict[id][0], self.whdict[id][1], self.id_count]