#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

class ThreadPool
{
public:
    ThreadPool(size_t numThreads)
        : stop(false)
    {
        // Instanciate all threads
        printf("Instantiated: %d threads\n", (int)numThreads);
        for (size_t i = 0; i < numThreads; ++i)
        {
            workers.emplace_back([this]
            {
                while (true)
                {
                    std::function<void()> task;
                    {
                        // Wait an awake/stop command
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this]
                        {
                            return this->stop || !this->tasks.empty();
                        });

                        // Check if not requested to stop
                        if (this->stop && this->tasks.empty())
                            return;
                        
                        // Take the first task to run
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template<class Func, class... Args>
    auto enqueue(Func&& func, Args&&... args) -> std::future<decltype(func(args...))>
    {
        using ReturnType = decltype(func(args...));

        // Build the task
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

        std::future<ReturnType> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }

        // Send awake command
        condition.notify_one();
        return res;
    }

    ~ThreadPool()
    {
        // Send stop command
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        // Waits all threads to end
        condition.notify_all();
        for (std::thread& worker : workers)
            worker.join();
    }
    
    inline const size_t GetTasksCount() const
    {
        return tasks.size();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};