class TSP:
    def __init__(self, file_path):
        with open(file_path, 'rt') as f:
            tsp_str = f.read()
        lines = tsp_str.strip().split('\n')

        parsing_coords = False

        self.node_coord_section = []

        for line in lines:
            line = line.strip()
            if not line or line == "EOF":
                continue

            if "NODE_COORD_SECTION" in line:
                parsing_coords = True
                continue

            # metadata parsing
            if not parsing_coords:
                parts = [p.strip() for p in line.split(':')]
                if len(parts) == 2:
                    key, value = parts
                    if key == "NAME":
                        self.name = value
                    elif key == "COMMENT":
                        self.comment = value
                    elif key == "TYPE":
                        self.type = value
                    elif key == "DIMENSION":
                        self.dimension = value
                    elif key == "EDGE_WEIGHT_TYPE":
                        self.edge_weight_type = value
            else:
                parts = line.split()
                # x = int(parts[1]) if self.check_int(parts[1]) else float(parts[1])
                # y = int(parts[2]) if self.check_int(parts[2]) else float(parts[2])
                x = float(parts[1])
                y = float(parts[2])
                self.node_coord_section.append((x,y))

    def check_int(self, num_str):
        if num_str.isdecimal():
            return True
        return False

    def print_data(self):
        print(self.name)
        print(self.comment)
        print(self.type)
        print(self.dimension)
        print(self.edge_weight_type)
        print(len(self.node_coord_section))
        print(self.node_coord_section[:10])

if __name__ == "__main__":
    tsp = TSP('mona-lisa100k.tsp')
    tsp.print_data()