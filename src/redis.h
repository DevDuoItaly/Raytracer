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

        // Clear all image entries
        redisReply* reply = (redisReply*) redisCommand(m_Context, "DEL image");
        freeReplyObject(reply);

        return m_Context->err;
    }

    inline void SendImage(unsigned char* image, int x, int y, int width, int height, int elementSize)
    {
        unsigned char coords[2 * sizeof(int)];
        memcpy(coords,               &x, sizeof(int));
        memcpy(coords + sizeof(int), &y, sizeof(int));
        redisReply* reply = (redisReply*) redisCommand(m_Context, "LPUSH image %b%b", coords, 2 * sizeof(int), image, width * height * elementSize);
        // printf("RESPONSE: %s\n", reply->str);

        freeReplyObject(reply);
    }

    inline void ReceiveImage(unsigned char* image, int& x, int& y, int width, int height, int elementSize)
    {
        redisReply* reply = (redisReply*) redisCommand(m_Context, "RPOP image");
        // printf("RESPONSE: %s\n", reply->str);
        
        memcpy(&x, reply->str,               sizeof(int));
        memcpy(&y, reply->str + sizeof(int), sizeof(int));
        memcpy(image, reply->str + 2 * sizeof(int), width * height * elementSize);
        
        freeReplyObject(reply);
    }

    inline int GetCount()
    {
        redisReply* reply = (redisReply*) redisCommand(m_Context, "LLEN image");
        int len = (int)reply->integer;
        freeReplyObject(reply);

        return len;
    }

    inline void Free()
    {
        if(m_Context)
            redisFree(m_Context);
    }

private:
    redisContext* m_Context;
};
