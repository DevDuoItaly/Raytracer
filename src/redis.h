#pragma once

#include <hiredis/hiredis.h>
#include <cstring>

class Redis
{
public:
    Redis() {}

    inline ~Redis()
    {
        Free();
    }

    inline int Connect()
    {
        m_Context = redisConnect("localhost", 6379);
        return m_Context->err;
    }

    inline void SendImage(unsigned char* image, int width, int height, int elementSize)
    {
        redisReply* reply = (redisReply*) redisCommand(m_Context, "SET image %b", image, width * height * elementSize);
        // printf("RESPONSE: %s\n", reply->str);

        freeReplyObject(reply);
    }

    inline void ReceiveImage(unsigned char* image, int width, int height, int elementSize)
    {
        redisReply* reply = (redisReply*) redisCommand(m_Context, "GET image");
        // printf("RESPONSE: %s\n", reply->str);

        memcpy(image, reply->str, width * height * elementSize);
        
        freeReplyObject(reply);
    }

    inline void Free()
    {
        if(m_Context)
            redisFree(m_Context);
    }

private:
    redisContext* m_Context;
};
