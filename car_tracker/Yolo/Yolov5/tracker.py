import math


class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.up = 0
        self.down = 0
        self.counted_id = []
        self.rem_ids = []

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + w) // 2
            cy = (y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, ptl in self.center_points.items():
                  pt = ptl[-1]
                  dist = math.hypot(cx - pt[0], cy - pt[1])

                  if dist < 20:
                      self.center_points[id].append((cx, cy))
  #                    print(self.center_points)
                      objects_bbs_ids.append([x, y, w, h, id])
                      same_object_detected = True
                      break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = [(cx, cy)]
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

            for id in self.center_points.keys():
                coords = self.center_points[id]
                xc = [coord[0] for coord in coords]
                yc = [coord[1] for coord in coords]
                if len(yc) >= 2:
                  direction = "up" if yc[-1] < yc[-2] else "down"
                  yp = yc[-1]
                  xp = xc[-1]
                  if int(yp) in [i for i in range(320, 335)] and id not in self.counted_id and int(xp) in [i for i in range(135,340)]:
                      if direction == 'up':
                          self.up += 1
                      else:
                          self.down += 1
                      self.counted_id.append(id)
                
                if len(yc) >= 3:
                  direction_aux = "up" if yc[-1] < yc[-3] else "down"
                  if int(yp) in range(410, 425) and id not in self.rem_ids and direction_aux == 'up':
                    self.rem_ids.append(id)
                  elif int(yp) in range(110, 125) and id not in self.rem_ids and direction_aux == 'down':
                    self.rem_ids.append(id)



        """
        if self.id_count >= 3:
            try:
                print(self.center_points[3])
                if len(self.center_points[3]) > 30:
                    print("U")
            except:
                print("E")
        """
        return objects_bbs_ids