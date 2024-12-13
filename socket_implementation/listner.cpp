#include <iostream>
#include <string>
#include <sstream>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <thread>

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(65432);

    // Bind socket
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    listen(server_fd, 3);

    std::cout << "Waiting for connection...\n";
    new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
    if (new_socket < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "Connected.\n";

    char buffer[1024] = {0};
    while (true) {
        int valread = read(new_socket, buffer, 1024);
        if (valread > 0) {
            std::string data(buffer, valread);
            std::stringstream ss(data);
            int x, y;
            char delimiter;

            // Parse coordinates
            ss >> x >> delimiter >> y;
            std::cout << "Received coordinates: (" << x << ", " << y << ")\n";

            // Hardware Logic Implementation 
        }
        // Wait for 1 second
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    return 0;
}